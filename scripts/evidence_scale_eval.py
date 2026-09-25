"""Is the out-of-distribution miscalibration "too much evidence", or a shifted intercept?

Two one-parameter corrections, both fitted on inner-val, both applied to test:

  temperature   z -> z/T with S = alpha+beta held fixed (spec §10, already run)
  evidence      e -> c*e for every head, i.e. alpha = c*sum(e1) + 1,
                beta = c*sum(e0) + 1 — the model is told it has less evidence
                than it thinks

They are NOT the same knob. Temperature is a monotone map of the fused logit,
so it cannot move AUC. Evidence scaling is not: the +1 of the Beta prior does
not scale, so a subject with little evidence is pulled toward 0.5 harder than a
confident one. Ranking can change, and the size of that change is reported —
it is the part of the story temperature cannot express.

Neither can express a pure base-rate shift, so the prior oracle (the offset from
the OBSERVED test prevalence, an upper bound and not deployable) is carried
along as the yardstick: whatever it fixes and the other two do not is intercept.

    uv run python scripts/evidence_scale_eval.py
    uv run python scripts/evidence_scale_eval.py --json results/evidence_scale.json

Pure arithmetic on runs/*/predictions.csv — no inference, no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from aggregate_results import SPLIT_KIND  # noqa: E402
from openidh_model.eval.metrics import compute_metrics  # noqa: E402
from openidh_model.train.calibrate import apply_affine, fit_temperature, prior_offset  # noqa: E402
from openidh_model.train.monitor import CLASSIFIERS  # noqa: E402

_EPS = 1e-7


def evidence_sums(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(sum e1, sum e0) over the heads — alpha - 1 and beta - 1."""
    e1 = sum(df[f"e1_{h}"].to_numpy(float) for h in CLASSIFIERS)
    e0 = sum(df[f"e0_{h}"].to_numpy(float) for h in CLASSIFIERS)
    return e1, e0


def scale_evidence(e1, e0, c: float):
    return c * e1 + 1.0, c * e0 + 1.0


def fit_scale(e1, e0, y, bounds=(1e-3, 1e3)) -> float:
    """c > 0 minimising inner-val NLL. c < 1 means "you have less evidence than
    you think"; c > 1 the opposite."""
    yv = np.asarray(y, float)

    def nll(log_c: float) -> float:
        a, b = scale_evidence(e1, e0, float(np.exp(log_c)))
        p = np.clip(a / (a + b), _EPS, 1 - _EPS)
        return float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))

    r = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(r.x))


def eval_run(run_dir: Path) -> dict | None:
    pf, rj = run_dir / "predictions.csv", run_dir / "result.json"
    if not (pf.is_file() and rj.is_file()):
        return None
    res = json.loads(rj.read_text())
    if res.get("train_yaml") != "train.yaml" or res["split_file"] not in SPLIT_KIND:
        return None
    label, kind = SPLIT_KIND[res["split_file"]]
    if res.get("fold_id") not in (None, ""):
        label = f"{label} f{res['fold_id']}"

    df = pd.read_csv(pf)
    v, t = df[df.role == "val"], df[df.role == "test"]
    ve1, ve0 = evidence_sums(v)
    te1, te0 = evidence_sums(t)
    vy, ty = v["y_true"].to_numpy(float), t["y_true"].to_numpy(float)
    va, vb = ve1 + 1, ve0 + 1
    ta, tb = te1 + 1, te0 + 1
    err = max(np.abs(ta - t["alpha"]).max(), np.abs(tb - t["beta"]).max())
    assert err < 1e-3, f"{run_dir.name}: evidence sum does not reproduce alpha/beta ({err:.2e})"

    T = fit_temperature(va, vb, vy)
    c = fit_scale(ve1, ve0, vy)
    off = prior_offset(float(vy.mean()), float(ty.mean()))   # oracle: uses the test prevalence

    raw = compute_metrics(ta, tb, ty)
    temp = compute_metrics(*apply_affine(ta, tb, T=T), ty)
    sc_a, sc_b = scale_evidence(te1, te0, c)
    scaled = compute_metrics(sc_a, sc_b, ty)
    both = compute_metrics(*apply_affine(sc_a, sc_b, offset=off), ty)
    oracle = compute_metrics(*apply_affine(ta, tb, offset=off), ty)

    return {
        "run": run_dir.name, "unit": label, "kind": kind, "seed": res["seed"],
        "T": T, "c": c, "prior_offset": off,
        "val_base_rate": float(vy.mean()), "test_base_rate": float(ty.mean()),
        "median_S_raw": float(np.median(ta + tb)), "median_S_scaled": float(np.median(sc_a + sc_b)),
        "raw": raw, "temperature": temp, "evidence_scaled": scaled,
        "prior_oracle": oracle, "scaled_plus_prior": both,
    }


def table(rows: list[dict]) -> str:
    df = pd.DataFrame([{**{k: r[k] for k in ("run", "unit", "kind", "c", "T", "prior_offset",
                                             "median_S_raw", "median_S_scaled")},
                        **{f"{m}_{k}": r[m][k] for m in ("raw", "temperature", "evidence_scaled",
                                                         "prior_oracle", "scaled_plus_prior")
                           for k in ("ece", "nll", "auc")}} for r in rows])
    g = df.groupby(["unit", "kind"], observed=True).mean(numeric_only=True).reset_index()
    g = g.sort_values(["kind", "unit"], ascending=[True, True])

    out = ["| ユニット | 条件 | c | median S 生→縮小 | ECE 生 | ECE 温度 | ECE 証拠縮小 | ECE prior(oracle) | ECE 縮小+prior | ΔAUC(縮小) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        out.append(
            f"| {r['unit']} | {'分布外' if r['kind'] == 'OOD' else '分布内'} | {r['c']:.2f} | "
            f"{r['median_S_raw']:.0f} → {r['median_S_scaled']:.0f} | {r['raw_ece']:.3f} | "
            f"{r['temperature_ece']:.3f} | {r['evidence_scaled_ece']:.3f} | {r['prior_oracle_ece']:.3f} | "
            f"{r['scaled_plus_prior_ece']:.3f} | {r['evidence_scaled_auc'] - r['raw_auc']:+.3f} |")

    out += ["", "| グループ | ECE 生 | 温度 | 証拠縮小 | prior(oracle) | 縮小+prior | NLL 生 → 縮小 | ΔAUC(縮小) |",
            "|---|---|---|---|---|---|---|---|"]
    for kind, lab in (("OOD", "分布外"), ("in-dist", "分布内")):
        d = df[df.kind == kind]
        if not len(d):
            continue
        out.append(f"| {lab} (n={len(d)}) | {d['raw_ece'].mean():.3f} | {d['temperature_ece'].mean():.3f} | "
                   f"{d['evidence_scaled_ece'].mean():.3f} | {d['prior_oracle_ece'].mean():.3f} | "
                   f"{d['scaled_plus_prior_ece'].mean():.3f} | "
                   f"{d['raw_nll'].mean():.3f} → {d['evidence_scaled_nll'].mean():.3f} | "
                   f"{(d['evidence_scaled_auc'] - d['raw_auc']).mean():+.3f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs_dir = Path(a.runs_dir)
    names = a.runs or sorted(d.name for d in runs_dir.iterdir() if (d / "predictions.csv").is_file())

    rows = [r for r in (eval_run(runs_dir / n) for n in names) if r is not None]
    if not rows:
        raise SystemExit("no production run with predictions.csv found")
    print(f"{len(rows)} runs\n")
    print(table(rows))
    d = pd.DataFrame(rows)
    print("\nc の分布:", " ".join(f"{r['unit']}:{r['c']:.2f}" for r in rows))
    print("注意: 証拠縮小は単調変換ではない（Beta の +1 が縮まないため）ので AUC が動きうる。上表の ΔAUC を参照。")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=1, default=float))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
