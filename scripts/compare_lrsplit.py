"""Baseline vs separated learning rates, on the metrics the experiment was run for.

Pairs runs/<tag>_s<seed> against runs/<variant>_<tag>_s<seed> and reports the
before/after table: tabular evidence (the thing the experiment targets), AUC, ECE,
ECE after the age-based intercept, best epoch, sensitivity at 0.5, and whether the
uncertainty signal survives.

    uv run python scripts/compare_lrsplit.py --variant lrsplit1
    uv run python scripts/compare_lrsplit.py --variant lrsplit3 --tag foldA
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

from age_prior_eval import _fit_age_model, _lo, subject_table  # noqa: E402
from openidh_model.eval.metrics import compute_metrics  # noqa: E402
from openidh_model.train.calibrate import apply_affine  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402

SPLIT = {"foldA": "splits_loso_foldA.csv", "foldB": "splits_loso_foldB.csv"}


def _age_offset(paths, subj, split_file, ages_test):
    """Zero-label intercept from the age model — recomputed per fold, not per run."""
    sp = pd.read_csv(paths.splits_dir / split_file).merge(
        subj[["subject_id", "age"]], on="subject_id", how="left")
    sp["mut"] = (sp["idh"].astype(str).str.strip() == "Mut").astype(int)
    tr = sp[sp.split_role == "train"].dropna(subset=["age"])
    va = sp[sp.split_role == "val"].dropna(subset=["age"])
    pi_of = _fit_age_model(tr["age"].to_numpy(float), tr["mut"].to_numpy(int), False)
    return _lo(float(pi_of(ages_test).mean())) - _lo(float(va["mut"].mean()))


def summarise(run_dir: Path, paths, subj, split_file) -> dict | None:
    rj, pf = run_dir / "result.json", run_dir / "predictions.csv"
    if not rj.is_file():
        return None
    r = json.loads(rj.read_text())
    mon = r["test_monitor"]
    out = {"run": run_dir.name, "auc": r["test_metrics"]["auc"], "ece": r["test_metrics"]["ece"],
           "best_epoch": r["best_epoch"], "ev_tabular": mon["evidence_balance"]["tabular"],
           "ev_image": float(np.mean([mon["evidence_balance"][k]
                                      for k in ("T1", "T2", "FLAIR", "T1c", "unified")])),
           "unc_ok": mon["uncertainty_ok"],
           "S_correct": mon["S_correct_median"], "S_wrong": mon["S_wrong_median"]}
    if pf.is_file():
        t = pd.read_csv(pf)
        t = t[t.role == "test"].merge(subj[["subject_id", "age"]], on="subject_id", how="left")
        a, b, y = t["alpha"].to_numpy(), t["beta"].to_numpy(), t["y_true"].to_numpy(float)
        p = a / (a + b)
        out["sens"] = float(((p > .5) & (y == 1)).sum() / max((y == 1).sum(), 1))
        out["spec"] = float(((p <= .5) & (y == 0)).sum() / max((y == 0).sum(), 1))
        off = _age_offset(paths, subj, split_file, t["age"].to_numpy(float))
        aa, bb = apply_affine(a, b, offset=off)
        out["ece_age"] = compute_metrics(aa, bb, y)["ece"]
        out["offset_age"] = off
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--variant", default="lrsplit1")
    ap.add_argument("--tag", default="foldB")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--paths-file", default=None)
    a = ap.parse_args()

    paths = load_paths(a.paths_file or "configs/paths.local.yaml")
    subj = subject_table(paths)
    rd, sf = Path(a.runs_dir), SPLIT[a.tag]

    base = [summarise(rd / f"{a.tag}_s{s}", paths, subj, sf) for s in a.seeds]
    new = [summarise(rd / f"{a.variant}_{a.tag}_s{s}", paths, subj, sf) for s in a.seeds]
    base, new = [x for x in base if x], [x for x in new if x]
    if not new:
        raise SystemExit(f"no {a.variant}_{a.tag}_s* runs under {rd}/ — the job has not landed yet")
    print(f"{a.tag}: baseline {len(base)} run(s) vs {a.variant} {len(new)} run(s)\n")

    def g(rows, k):
        v = [x[k] for x in rows if k in x]
        return float(np.mean(v)) if v else float("nan")

    print("| 項目 | 変更前 | 変更後 | 変化 |")
    print("|---|---|---|---|")
    for k, nm, fmt, want in (
        ("ev_tabular", "tabular evidence", "{:.2f}", "up"),
        ("ev_image", "画像ヘッド evidence（平均）", "{:.2f}", None),
        ("auc", "AUC", "{:.3f}", "up"),
        ("ece", "ECE", "{:.3f}", "down"),
        ("ece_age", "年齢ベース切片後の ECE", "{:.3f}", "down"),
        ("sens", "感度 @0.5", "{:.1%}", "up"),
        ("spec", "特異度 @0.5", "{:.1%}", None),
        ("best_epoch", "ベスト epoch", "{:.1f}", None),
        ("S_correct", "S（正答）", "{:.1f}", None),
        ("S_wrong", "S（誤答）", "{:.1f}", None),
    ):
        o, n = g(base, k), g(new, k)
        if np.isnan(o) and np.isnan(n):
            continue
        d = n - o
        arrow = "" if want is None else (" ✅" if (d > 0) == (want == "up") else " ⚠️")
        print(f"| {nm} | {fmt.format(o)} | {fmt.format(n)} | {d:+.3f}{arrow} |")

    ok_b = sum(1 for x in base if x["unc_ok"]), len(base)
    ok_n = sum(1 for x in new if x["unc_ok"]), len(new)
    print(f"\n誤答時 S < 正答時 S: 変更前 {ok_b[0]}/{ok_b[1]} → 変更後 {ok_n[0]}/{ok_n[1]}")
    if any("offset_age" in x for x in new):
        print(f"年齢ベースの推定オフセット: {g(new, 'offset_age'):+.2f}"
              f"（学習設定に依存しないので変更前と同じはず: {g(base, 'offset_age'):+.2f}）")


if __name__ == "__main__":
    main()
