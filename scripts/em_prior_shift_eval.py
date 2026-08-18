"""Zero-shot calibration: can EM recover the target prevalence without labels? (spec §10)

The N-case curve gave a prescription that costs labels. This asks whether the same
scalar can be had for free: Saerens-Latinne-Decaestecker EM estimates the target
prevalence from the *shape* of the unlabelled predicted-probability distribution,
which — given that the miscalibration is very nearly a constant logit offset — is
the only parameter that needs estimating.

    uv run python scripts/em_prior_shift_eval.py
    uv run python scripts/em_prior_shift_eval.py --unit "LOSO-B" --reps 50

Reports, per fold-unit: ECE raw / temperature / EM / labelled-intercept, the
estimated vs true prevalence, the estimated vs observed logit offset, how the
estimate behaves with fewer unlabelled cases, and the AUC/AUPRC invariance check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from aggregate_results import load_runs  # noqa: E402
from openidh_model.eval.metrics import compute_metrics  # noqa: E402
from openidh_model.train.calibrate import (  # noqa: E402
    apply_affine, check_ranking_invariant, em_prior_shift, fit_temperature, prior_offset,
)
from sample_size_curve import _fit_offset  # noqa: E402

NS = (20, 50, 100, 200)


def _load(runs_dir: Path):
    meta = load_runs(runs_dir)
    frames = []
    for r in meta:
        f = runs_dir / r["run"] / "predictions.csv"
        if f.is_file():
            frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit("no predictions.csv — run scripts/calibrate_runs.py first")
    return meta, pd.concat(frames, ignore_index=True)


def per_run(pred: pd.DataFrame, run: str, reps: int, rng) -> dict:
    d = pred[pred["run"] == run]
    v, t = d[d["role"] == "val"], d[d["role"] == "test"]
    va, vb, vy = v["alpha"].to_numpy(), v["beta"].to_numpy(), v["y_true"].to_numpy(float)
    ta, tb, ty = t["alpha"].to_numpy(), t["beta"].to_numpy(), t["y_true"].to_numpy(float)
    p_test = ta / (ta + tb)
    pi_src, pi_true = float(vy.mean()), float(ty.mean())

    em = em_prior_shift(p_test, pi_src)
    T = fit_temperature(va, vb, vy)
    off_lab = _fit_offset(ta, tb, ty, pi_src, "nll")      # labelled intercept (lower bound)

    def ece(off=0.0, T_=1.0):
        return compute_metrics(*apply_affine(ta, tb, T=T_, offset=off), ty)["ece"]

    # how many unlabelled cases does EM need?
    subs = []
    for n in NS:
        if len(p_test) <= n:
            continue
        est = [em_prior_shift(p_test[rng.permutation(len(p_test))[:n]], pi_src)["pi_hat"]
               for _ in range(reps)]
        subs.append({"n": n, "pi_mean": float(np.mean(est)), "pi_sd": float(np.std(est, ddof=1))})

    return {
        "run": run, "pi_source": pi_src, "pi_true": pi_true, "pi_em": em["pi_hat"],
        "em_converged": em["converged"], "em_iters": em["n_iter"], "em_tail_range": em["tail_range"],
        "offset_em": em["offset"], "offset_prior_oracle": prior_offset(pi_src, pi_true),
        "offset_labelled": off_lab,
        "ece_raw": ece(), "ece_temp": ece(T_=T), "ece_em": ece(off=em["offset"]),
        "ece_labelled": ece(off=off_lab),
        "ranking_delta_em": check_ranking_invariant(ta, tb, ty, offset=em["offset"]),
        "subsample": subs, "n_test": len(ty),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--unit", default=None, help="one unit label (default: all)")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    meta, pred = _load(Path(a.runs_dir))
    rng = np.random.default_rng(0)
    rows = [dict(per_run(pred, r["run"], a.reps, rng), label=r["label"], kind=r["kind"])
            for r in meta if a.unit is None or r["label"] == a.unit]

    units, order = {}, []
    for r in rows:
        units.setdefault(r["label"], []).append(r)
        if r["label"] not in order:
            order.append(r["label"])
    order.sort(key=lambda l: (units[l][0]["kind"] != "OOD", l))
    m = lambda rs, k: float(np.mean([x[k] for x in rs]))  # noqa: E731

    print("\n## 1. ECE: 生 / temperature / EM / ラベル付き切片\n")
    print("| ユニット | 条件 | ECE 生 | ECE temp | ECE EM | ECE 全例切片 | EM の改善 |")
    print("|---|---|---|---|---|---|---|")
    for lab in order:
        rs = units[lab]
        er, et, ee, el = (m(rs, k) for k in ("ece_raw", "ece_temp", "ece_em", "ece_labelled"))
        print(f"| {lab} | {'分布外' if rs[0]['kind'] == 'OOD' else '分布内'} | {er:.3f} | {et:.3f} | "
              f"{ee:.3f} | {el:.3f} | {(er - ee) / er * 100:+.0f}% |")
    for nm, sel in (("分布外", "OOD"), ("分布内", "in-dist")):
        rs = [r for r in rows if r["kind"] == sel]
        if rs:
            print(f"| **{nm}** | | {m(rs,'ece_raw'):.3f} | {m(rs,'ece_temp'):.3f} | "
                  f"{m(rs,'ece_em'):.3f} | {m(rs,'ece_labelled'):.3f} | "
                  f"{(m(rs,'ece_raw') - m(rs,'ece_em')) / m(rs,'ece_raw') * 100:+.0f}% |")

    print("\n## 2. EM が推定した有病率 vs 真の有病率\n")
    print("| ユニット | 訓練時 (inner-val) | 真の値 (test) | EM 推定 | 誤差 | 収束 |")
    print("|---|---|---|---|---|---|")
    for lab in order:
        rs = units[lab]
        ps, pt, pe = m(rs, "pi_source"), m(rs, "pi_true"), m(rs, "pi_em")
        conv = sum(1 for x in rs if x["em_converged"])
        print(f"| {lab} | {ps*100:.1f}% | {pt*100:.1f}% | {pe*100:.1f}% | {(pe-pt)*100:+.1f}pt | "
              f"{conv}/{len(rs)} |")

    print("\n## 3. オフセット: EM 推定 vs base rate オラクル vs ラベル付き\n")
    print("| ユニット | EM | prior オラクル | ラベル付き切片 |")
    print("|---|---|---|---|")
    for lab in order:
        rs = units[lab]
        print(f"| {lab} | {m(rs,'offset_em'):+.2f} | {m(rs,'offset_prior_oracle'):+.2f} | "
              f"{m(rs,'offset_labelled'):+.2f} |")

    print("\n## 4. 推定に使うラベルなし症例数の影響（有病率の推定値）\n")
    print("| ユニット | " + " | ".join(f"n={n}" for n in NS) + " | 全例 | 真値 |")
    print("|---|" + "---|" * (len(NS) + 2))
    for lab in order:
        rs = units[lab]
        cells = []
        for n in NS:
            vals = [s for x in rs for s in x["subsample"] if s["n"] == n]
            cells.append(f"{np.mean([v['pi_mean'] for v in vals])*100:.1f}% ± {np.mean([v['pi_sd'] for v in vals])*100:.1f}"
                         if vals else "—")
        print(f"| {lab} | " + " | ".join(cells) + f" | {m(rs,'pi_em')*100:.1f}% | {m(rs,'pi_true')*100:.1f}% |")

    d = max(abs(r["ranking_delta_em"][k] or 0.0) for r in rows for k in ("auc", "auprc"))
    print(f"\n## 5. 検算: AUC / AUPRC の最大変化量 {d:.2e}（EM も単調変換なので 0 のはず）")
    bad = [r["run"] for r in rows if not r["em_converged"]]
    print(f"\n収束: {len(rows) - len(bad)}/{len(rows)} ラン" + (f"  未収束: {bad}" if bad else ""))

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2, default=float))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
