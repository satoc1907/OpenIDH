"""Would a per-case "distance from the source domain" be worth acting on?

The idea under test: score each incoming image by how far it sits from the
training distribution, and down-weight the heads when it is far. That is only
worth building if the distance carries information the model does not already
emit. Two questions, both answerable from the dumped CLS features
(runs/*/features_*.npz) plus predictions.csv — no inference, no GPU:

  1. does the distance separate a source case from a shifted case?   (per case)
  2. does it predict which cases the model gets wrong — better than the
     model's own evidence mass S, or than |p_hat - 0.5|?

The distance is deliberately the simplest defensible one: per-dimension
z-scores against the inner-val mean/sd of that encoder, RMS over the 384 dims,
averaged over the five image encoders. A fancier score (Mahalanobis with a
shrunk covariance, a trained OOD head) could do better, but the controls below
are what decide whether any of them pays off.

    uv run python scripts/ood_score_eval.py
    uv run python scripts/ood_score_eval.py --json results/ood_score.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openidh_model.eval import shift_report as sr  # noqa: E402
from openidh_model.eval.features import ENCODERS  # noqa: E402

RUNS = [f"{u}_s{s}" for u in ("foldA", "foldB", "field", "vendor") for s in range(3)] + ["rand0_s0"]


def distance(rd: dict, split: str) -> np.ndarray:
    """RMS z-score against the inner-val distribution, averaged over encoders."""
    out = []
    for e in ENCODERS:
        fv = rd["val"][f"feat_{e}"].astype(np.float64)
        f = rd[split][f"feat_{e}"].astype(np.float64)
        mu, sd = fv.mean(0), fv.std(0) + 1e-6
        out.append(np.sqrt((((f - mu) / sd) ** 2).mean(1)))
    return np.mean(out, 0)


def eval_run(runs_dir: Path, run: str) -> dict | None:
    rd = sr.load_run(runs_dir, run)
    pf = runs_dir / run / "predictions.csv"
    if rd is None or not pf.is_file():
        return None
    dv, dt = distance(rd, "val"), distance(rd, "test")
    p = pd.read_csv(pf)
    t = p[p.role == "test"]
    assert len(t) == len(dt), f"{run}: {len(t)} predictions vs {len(dt)} feature rows"

    y = t["y_true"].to_numpy(float)
    ph = t["p_hat"].to_numpy(float)
    S = t["S"].to_numpy(float)
    err = ((ph > 0.5).astype(int) != y.astype(int)).astype(int)
    auc = lambda s: float(roc_auc_score(err, s)) if err.any() and not err.all() else float("nan")  # noqa: E731
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)  # noqa: E731

    return {
        "run": run, "unit": sr.unit_of(run)[0], "n_val": len(dv), "n_test": len(dt),
        "d_val": float(dv.mean()), "d_test": float(dt.mean()),
        # 1. can one case be told apart from a source case?
        "auc_source_vs_shifted": float(roc_auc_score(
            np.r_[np.zeros(len(dv)), np.ones(len(dt))], np.r_[dv, dt])),
        # 2. what predicts an error, on the shifted set?
        "err_auc_distance": auc(dt),
        "err_auc_low_S": auc(-S),                      # the model's own evidence mass
        "err_auc_near_half": auc(-np.abs(ph - 0.5)),   # the model's own probability
        "err_auc_distance_plus_low_S": auc(z(dt) - z(S)),
        "err_rate": float(err.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs_dir = Path(a.runs_dir)

    rows = [r for r in (eval_run(runs_dir, n) for n in (a.runs or RUNS)) if r is not None]
    if not rows:
        raise SystemExit("no run has both features_*.npz and predictions.csv")
    df = pd.DataFrame(rows)
    cols = ["auc_source_vs_shifted", "err_auc_distance", "err_auc_low_S",
            "err_auc_near_half", "err_auc_distance_plus_low_S"]
    print(df[["run", "unit", "n_test", "err_rate"] + cols].round(3).to_string(index=False))
    print("\nunit 平均:")
    print(df.groupby("unit", observed=True)[cols].mean().round(3).to_string())
    print("\n読み方: 1 列目が 0.5 に近ければ「1 症例では分布外と判定できない」。"
          "\n        2 列目が 3〜4 列目を上回らなければ、距離はモデル自身の出力に情報を足していない。")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=1, default=float))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
