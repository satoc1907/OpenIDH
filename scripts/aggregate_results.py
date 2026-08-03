"""Aggregate finished runs/ into the tables needed to judge the model (spec §15).

Reads every runs/*/result.json, groups the 9 fold-units into OOD (site / vendor /
field shift) vs in-distribution (random 5-fold), and reports discrimination,
calibration, uncertainty health, evidence balance and early-stopping behaviour.

    uv run python scripts/aggregate_results.py
    uv run python scripts/aggregate_results.py --runs-dir runs --csv results/runs.csv

Smoke/probe runs are skipped by default: only runs whose train_yaml is train.yaml
count as production (--include-all to override).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# split_file -> (short label, shift kind). fold_id is appended for the 5-fold file.
SPLIT_KIND = {
    "splits_loso_foldA.csv": ("LOSO-A", "OOD"),
    "splits_loso_foldB.csv": ("LOSO-B", "OOD"),
    "splits_vendor_philips.csv": ("vendor Philips", "OOD"),
    "splits_field.csv": ("field 3T", "OOD"),
    "splits_random_5fold.csv": ("random", "in-dist"),
}
HEADS = ["T1", "T2", "FLAIR", "T1c", "unified", "tabular"]


def load_runs(runs_dir: Path, include_all: bool = False) -> list[dict]:
    rows = []
    for rj in sorted(runs_dir.glob("*/result.json")):
        d = json.loads(rj.read_text())
        if not include_all and d.get("train_yaml") != "train.yaml":
            continue
        split = d["split_file"]
        if split not in SPLIT_KIND:
            continue
        label, kind = SPLIT_KIND[split]
        if d.get("fold_id") not in (None, ""):
            label = f"{label} f{d['fold_id']}"
        m, mon = d["test_metrics"], d["test_monitor"]
        hist = json.loads((rj.parent / "history.json").read_text())
        rows.append(dict(
            run=rj.parent.name, label=label, kind=kind, seed=d["seed"],
            n_train=d["n_train"], n_val=d["n_val"], n_test=d["n_test"],
            epochs_run=len(hist), best_epoch=d["best_epoch"], best_val_nll=d["best_val_nll"],
            **{k: m[k] for k in ("auc", "auprc", "ece", "brier", "nll", "mean_S", "median_S")},
            S_correct=mon["S_correct_median"], S_wrong=mon["S_wrong_median"],
            unc_ok=mon["uncertainty_ok"], reg_over_data=mon["reg_over_data"],
            **{f"ev_{h}": mon["evidence_balance"].get(h, float("nan")) for h in HEADS},
        ))
    return rows


def _ms(rows: list[dict], key: str) -> tuple[float, float]:
    """mean, sample sd (0 for a single run)."""
    v = np.array([r[key] for r in rows], dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def _by_label(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    order, seen = [], set()
    for r in sorted(rows, key=lambda r: (r["kind"] != "OOD", r["label"])):
        if r["label"] not in seen:
            seen.add(r["label"]); order.append(r["label"])
    return [(lab, [r for r in rows if r["label"] == lab]) for lab in order]


def report(rows: list[dict]) -> None:
    W = 96
    print("=" * W)
    print(f"1. PER-UNIT  (mean +- sd over seeds; {len(rows)} runs)")
    print("=" * W)
    print(f"{'unit':17s}{'kind':9s}{'seeds':>6s}{'n_test':>7s}"
          f"{'AUC':>16s}{'AUPRC':>16s}{'ECE':>15s}{'Brier':>15s}")
    for lab, rs in _by_label(rows):
        cells = "".join(f"{m:>9.3f}+-{s:<5.3f}" for m, s in
                        (_ms(rs, k) for k in ("auc", "auprc", "ece", "brier")))
        print(f"{lab:17s}{rs[0]['kind']:9s}{len(rs):>6d}{rs[0]['n_test']:>7d}{cells}")

    ood = [r for r in rows if r["kind"] == "OOD"]
    ind = [r for r in rows if r["kind"] == "in-dist"]
    if ood and ind:
        print(f"\n{'=' * W}\n2. OOD vs IN-DISTRIBUTION  (headline robustness claim)\n{'=' * W}")
        print(f"{'metric':12s}{f'OOD ({len(ood)} runs)':>22s}"
              f"{f'in-dist ({len(ind)} runs)':>24s}{'gap':>12s}")
        for k in ("auc", "auprc", "ece", "brier", "nll"):
            mo, so = _ms(ood, k); mi, si = _ms(ind, k)
            print(f"{k:12s}{mo:>13.3f} +- {so:<5.3f}{mi:>15.3f} +- {si:<5.3f}{mo - mi:>+12.3f}")

    print(f"\n{'=' * W}\n3. UNCERTAINTY HEALTH  (S_wrong must sit BELOW S_correct)\n{'=' * W}")
    print(f"{'run':18s}{'S_correct':>11s}{'S_wrong':>10s}{'delta':>9s}{'ok':>6s}{'reg/data':>10s}")
    for r in rows:
        print(f"{r['run']:18s}{r['S_correct']:>11.1f}{r['S_wrong']:>10.1f}"
              f"{r['S_wrong'] - r['S_correct']:>+9.1f}{'OK' if r['unc_ok'] else 'WARN':>6s}"
              f"{r['reg_over_data']:>10.2f}")
    bad = [r["run"] for r in rows if not r["unc_ok"]]
    print(f"\nuncertainty_ok: {len(rows) - len(bad)}/{len(rows)}   failures: {bad or 'none'}")

    print(f"\n{'=' * W}\n4. EVIDENCE BALANCE  (mean scaled evidence per head)\n{'=' * W}")
    print(f"{'unit':17s}" + "".join(f"{h:>10s}" for h in HEADS))
    for lab, rs in _by_label(rows):
        print(f"{lab:17s}" + "".join(f"{_ms(rs, f'ev_{h}')[0]:>10.2f}" for h in HEADS))
    print(f"{'ALL':17s}" + "".join(f"{_ms(rows, f'ev_{h}')[0]:>10.2f}" for h in HEADS))

    print(f"\n{'=' * W}\n5. EARLY STOPPING\n{'=' * W}")
    e = np.array([r["epochs_run"] for r in rows]); b = np.array([r["best_epoch"] for r in rows])
    print(f"epochs run : min={e.min()} max={e.max()} mean={e.mean():.1f}")
    print(f"best epoch : min={b.min()} max={b.max()} mean={b.mean():.1f}")
    print(f"runs that ran the full 200 epochs: {(e >= 200).sum()}")

    print(f"\n{'=' * W}\n6. SEED SPREAD (AUC range per unit)\n{'=' * W}")
    for lab, rs in _by_label(rows):
        v = [r["auc"] for r in rs]
        print(f"{lab:17s}{rs[0]['kind']:9s}AUC {min(v):.3f} .. {max(v):.3f}   "
              f"spread {max(v) - min(v):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--csv", default=None, help="also write one tidy row per run here")
    ap.add_argument("--include-all", action="store_true",
                    help="include smoke/probe runs (default: train.yaml runs only)")
    a = ap.parse_args()

    rows = load_runs(Path(a.runs_dir), a.include_all)
    if not rows:
        raise SystemExit(f"no production runs found under {a.runs_dir}/")
    report(rows)

    if a.csv:
        out = Path(a.csv); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
