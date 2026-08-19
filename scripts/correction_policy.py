"""When is it safe to apply the offset correction at all? (deployment rule)

The N-case curve reports the average gain. A portal needs the other number: how
often the correction makes things *worse*. With N small the estimated offset is
noisy, and an offset estimated with the wrong sign actively harms — EM on LOSO-A
is the worked example (ECE 0.082 -> 0.141).

So for each N this measures, over bootstrap replicates:
  p_harm     share of replicates where corrected ECE > uncorrected ECE
  abstain    share where a deployable rule declines to correct
  p_harm_r   share of harmful replicates once that rule is applied

The rule uses only what a deployment has: resample the N labelled cases with
replacement, refit the offset, and correct only when |offset| exceeds k standard
errors of its own bootstrap distribution.

    uv run python scripts/correction_policy.py --unit "LOSO-B" --k 2
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
from openidh_model.train.calibrate import apply_affine, to_logit  # noqa: E402

NS = (10, 20, 50, 100, 200)
_GRID = np.linspace(-4, 4, 801)


def _fit(z, y):
    p = np.clip(1.0 / (1.0 + np.exp(-(z[None, :] + _GRID[:, None]))), 1e-7, 1 - 1e-7)
    nll = -(y[None, :] * np.log(p) + (1 - y[None, :]) * np.log(1 - p)).mean(1)
    return float(_GRID[int(np.argmin(nll))])


def _ece(a, b, y, off=0.0):
    return compute_metrics(*apply_affine(a, b, offset=off), y)["ece"]


def policy(pred: pd.DataFrame, runs: list[str], ns=NS, reps: int = 50,
           k: float = 2.0, inner: int = 25, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for n in ns:
        harm, harm_r, abst, gain, gain_r, ses = [], [], [], [], [], []
        for run in runs:
            d = pred[(pred["run"] == run) & (pred["role"] == "test")]
            a, b, y = d["alpha"].to_numpy(), d["beta"].to_numpy(), d["y_true"].to_numpy(float)
            z = to_logit(a, b)
            if len(y) <= n + 20:
                continue
            for _ in range(reps):
                idx = rng.permutation(len(y))
                f, ev = idx[:n], idx[n:]
                if len(np.unique(y[f])) < 2:
                    continue
                off = _fit(z[f], y[f])
                # deployable uncertainty: bootstrap inside the N labelled cases
                bs = [_fit(z[f][j], y[f][j]) for j in
                      (rng.integers(0, n, n) for _ in range(inner))]
                se = float(np.std(bs, ddof=1))
                ses.append(se)
                raw = _ece(a[ev], b[ev], y[ev])
                cor = _ece(a[ev], b[ev], y[ev], off)
                apply_it = abs(off) > k * se if se > 0 else True
                harm.append(cor > raw)
                gain.append(raw - cor)
                abst.append(not apply_it)
                eff = cor if apply_it else raw
                harm_r.append(eff > raw)
                gain_r.append(raw - eff)
        if not harm:
            continue
        out.append({
            "n": n, "reps": len(harm),
            "p_harm": float(np.mean(harm)), "p_harm_ruled": float(np.mean(harm_r)),
            "abstain": float(np.mean(abst)),
            "gain_mean": float(np.mean(gain)), "gain_ruled": float(np.mean(gain_r)),
            # how bad it gets when it does go wrong, not just how often
            "loss_when_harm": float(-np.mean([g for g in gain if g < 0])) if any(g < 0 for g in gain) else 0.0,
            "se_mean": float(np.mean(ses)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--unit", default="LOSO-B")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    meta = [r for r in load_runs(Path(a.runs_dir)) if r["label"] == a.unit]
    if not meta:
        raise SystemExit(f"unit {a.unit!r} not found")
    names = [r["run"] for r in meta]
    pred = pd.concat([pd.read_csv(Path(a.runs_dir) / n / "predictions.csv") for n in names],
                     ignore_index=True)
    rows = policy(pred, names, reps=a.reps, k=a.k)

    print(f"unit={a.unit}  rule: |offset| > {a.k}·SE(bootstrap)  reps/run={a.reps}\n")
    print(f"{'N':>5}{'P(悪化)':>10}{'ルール後':>10}{'見送り率':>10}"
          f"{'ECE 改善':>10}{'ルール後':>10}{'悪化時の幅':>11}{'SE':>8}")
    for r in rows:
        print(f"{r['n']:>5}{r['p_harm']*100:>9.0f}%{r['p_harm_ruled']*100:>9.0f}%"
              f"{r['abstain']*100:>9.0f}%{r['gain_mean']:>10.3f}{r['gain_ruled']:>10.3f}"
              f"{r['loss_when_harm']:>11.3f}{r['se_mean']:>8.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
