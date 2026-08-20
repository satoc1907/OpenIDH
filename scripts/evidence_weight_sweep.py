"""What would a louder head buy? Rescale one head's evidence post-hoc and re-fuse.

Fusion sums evidence, so a head's influence is set by the magnitude it emits, not
by how reliable it is. The tabular head is the case in point: it discriminates as
well as any image head (standalone AUC ~0.87) while emitting a fraction of their
evidence. This substitutes k x that head's evidence back into the fusion — exactly,
not approximately — and reports what the fused metrics would become.

It is a diagnostic, not a method: training at a higher head learning rate changes
what the head learns, not only how loud it is. What the sweep answers is whether
magnitude is the binding constraint before spending GPU hours finding out.

    uv run python scripts/evidence_weight_sweep.py --runs lrsplit1_foldB_s1
    uv run python scripts/evidence_weight_sweep.py --head unified --ks 0 1 2 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openidh_model.eval.metrics import compute_metrics  # noqa: E402

DEFAULT_KS = (0, 1, 2, 3, 5, 8, 12)


def sweep(pred: pd.DataFrame, head: str, ks=DEFAULT_KS) -> list[dict]:
    t = pred[pred["role"] == "test"]
    a, b = t["alpha"].to_numpy(float), t["beta"].to_numpy(float)
    y = t["y_true"].to_numpy(float)
    e1, e0 = t[f"e1_{head}"].to_numpy(float), t[f"e0_{head}"].to_numpy(float)
    out = []
    for k in ks:
        # swap this head's contribution for k x itself; the rest of the sum is untouched
        aa, bb = a - e1 + k * e1, b - e0 + k * e0
        m = compute_metrics(aa, bb, y)
        p = aa / (aa + bb)
        out.append({"k": k, "evidence": float((k * (e1 + e0)).mean()),
                    "auc": m["auc"], "ece": m["ece"], "nll": m["nll"],
                    "sens": float(((p > .5) & (y == 1)).sum() / max((y == 1).sum(), 1)),
                    "spec": float(((p <= .5) & (y == 0)).sum() / max((y == 0).sum(), 1))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--head", default="tabular")
    ap.add_argument("--ks", type=float, nargs="*", default=list(DEFAULT_KS))
    a = ap.parse_args()

    print(f"head={a.head}  （k=1 が実際の学習結果、k=0 はそのヘッドを外した場合）\n")
    print(f"{'run':22s}{'k':>5s}{'evidence':>10s}{'AUC':>8s}{'ECE':>8s}{'感度@.5':>9s}{'特異度':>8s}")
    for run in a.runs:
        f = Path(a.runs_dir) / run / "predictions.csv"
        if not f.is_file():
            print(f"{run}: predictions.csv なし — scripts/calibrate_runs.py を先に")
            continue
        for r in sweep(pd.read_csv(f), a.head, a.ks):
            print(f"{run:22s}{r['k']:>5g}{r['evidence']:>10.2f}{r['auc']:>8.3f}"
                  f"{r['ece']:>8.3f}{r['sens']*100:>8.1f}%{r['spec']*100:>7.1f}%")
        print()


if __name__ == "__main__":
    main()
