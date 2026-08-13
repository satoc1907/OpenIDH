"""How many labelled target-domain cases buy back the calibration? (spec §10 follow-up)

The prior-correction oracle showed that most of the out-of-distribution
miscalibration is a constant logit offset, but it used the *observed* test
prevalence — not something a deployment has. This asks the deployable version of
the question: given N labelled cases from the target site, estimate the offset
from those N alone and measure the calibration on everything else.

    uv run python scripts/sample_size_curve.py                  # LOSO-B
    uv run python scripts/sample_size_curve.py --unit "LOSO-A" --target-ece 0.08

Within each bootstrap replicate the test set is split into N fitting cases and an
evaluation remainder, and raw / N-estimated / oracle ECE are all measured on that
same remainder — so the three are comparable at every N.
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
from openidh_model.train.calibrate import apply_affine, prior_offset, to_logit  # noqa: E402

DEFAULT_NS = (10, 20, 50, 100, 200)


def _ece(a, b, y, offset=0.0):
    return compute_metrics(*apply_affine(a, b, offset=offset), y)["ece"]


def _fit_offset(a, b, y, pi_source, how: str):
    """Estimate the logit offset from a small labelled sample of the target domain."""
    if how == "prevalence":          # same estimator family as the oracle
        return prior_offset(pi_source, float(np.mean(y)))
    z = to_logit(a, b)               # 'nll': fit the intercept directly
    grid = np.linspace(-4, 4, 1601)
    p = 1.0 / (1.0 + np.exp(-(z[None, :] + grid[:, None])))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    nll = -(y[None, :] * np.log(p) + (1 - y[None, :]) * np.log(1 - p)).mean(1)
    return float(grid[int(np.argmin(nll))])


def curve(pred: pd.DataFrame, runs: list[str], pi_source: dict, ns=DEFAULT_NS,
          reps: int = 50, how: str = "prevalence", seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for n in ns:
        est, raw, ora, asy, offs = [], [], [], [], []
        for run in runs:
            d = pred[(pred["run"] == run) & (pred["role"] == "test")]
            a, b, y = d["alpha"].to_numpy(), d["beta"].to_numpy(), d["y_true"].to_numpy(float)
            if len(y) <= n + 20:
                continue
            oracle_off = prior_offset(pi_source[run], float(np.mean(y)))
            # asymptote inside the same estimator family: the offset this estimator
            # would reach given the whole target set
            full_off = _fit_offset(a, b, y, pi_source[run], how)
            for _ in range(reps):
                idx = rng.permutation(len(y))
                fit, ev = idx[:n], idx[n:]
                if len(np.unique(y[fit])) < 2:     # a single-class draw says nothing
                    continue
                off = _fit_offset(a[fit], b[fit], y[fit], pi_source[run], how)
                offs.append(off)
                est.append(_ece(a[ev], b[ev], y[ev], off))
                raw.append(_ece(a[ev], b[ev], y[ev], 0.0))
                ora.append(_ece(a[ev], b[ev], y[ev], oracle_off))
                asy.append(_ece(a[ev], b[ev], y[ev], full_off))
        e = np.array(est)
        if not len(e):   # N leaves too few cases to evaluate on in every run of this unit
            continue
        out.append({
            "n": n, "reps": len(e),
            "ece_mean": float(e.mean()), "ece_median": float(np.median(e)),
            "ece_lo": float(np.percentile(e, 2.5)), "ece_hi": float(np.percentile(e, 97.5)),
            "ece_raw": float(np.mean(raw)), "ece_oracle": float(np.mean(ora)),
            "ece_asymptote": float(np.mean(asy)),
            "offset_mean": float(np.mean(offs)), "offset_sd": float(np.std(offs, ddof=1)),
        })
    return out


def min_n_for(rows: list[dict], target: float, key: str = "ece_mean"):
    """Smallest N whose ECE clears the target (None if none of them do)."""
    ok = [r["n"] for r in rows if r[key] <= target]
    return min(ok) if ok else None


def build(runs_dir: Path, unit: str, ns=DEFAULT_NS, reps: int = 50, how: str = "prevalence"):
    meta = [r for r in load_runs(runs_dir) if r["label"] == unit]
    if not meta:
        raise SystemExit(f"unit {unit!r} not found")
    names = [r["run"] for r in meta]
    pf = [runs_dir / n / "predictions.csv" for n in names]
    if not all(p.is_file() for p in pf):
        raise SystemExit("predictions.csv missing — run scripts/calibrate_runs.py first")
    pred = pd.concat([pd.read_csv(p) for p in pf], ignore_index=True)
    pi_source = {n: float(pred[(pred["run"] == n) & (pred["role"] == "val")]["y_true"].mean())
                 for n in names}
    return {"unit": unit, "runs": names, "estimator": how, "reps_per_run": reps,
            "val_base_rate": pi_source,
            "test_base_rate": {n: float(pred[(pred["run"] == n) & (pred["role"] == "test")]
                                        ["y_true"].mean()) for n in names},
            "curve": curve(pred, names, pi_source, ns, reps, how)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--unit", default="LOSO-B")
    ap.add_argument("--ns", type=int, nargs="*", default=list(DEFAULT_NS))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--estimator", choices=("prevalence", "nll"), default="prevalence")
    ap.add_argument("--target-ece", type=float, default=0.08)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    res = build(Path(a.runs_dir), a.unit, tuple(a.ns), a.reps, a.estimator)
    rows = res["curve"]
    print(f"unit={a.unit}  runs={len(res['runs'])}  estimator={a.estimator}  reps/run={a.reps}")
    print(f"{'N':>5}{'ECE (mean)':>12}{'95% CI':>18}{'ECE raw':>10}{'ECE oracle':>12}{'asympt.':>10}{'offset':>16}")
    for r in rows:
        ci = f"[{r['ece_lo']:.3f}, {r['ece_hi']:.3f}]"
        off = f"{r['offset_mean']:+.2f} ± {r['offset_sd']:.2f}"
        print(f"{r['n']:>5}{r['ece_mean']:>12.3f}{ci:>18}"
              f"{r['ece_raw']:>10.3f}{r['ece_oracle']:>12.3f}{r['ece_asymptote']:>10.3f}{off:>16}")
    n = min_n_for(rows, a.target_ece)
    print(f"\nECE <= {a.target_ece}: " + (f"N >= {n}" if n else "どの N でも未達"))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
