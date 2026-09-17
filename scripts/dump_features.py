"""Dump CLS features + head weights for the shift decomposition (experiment A).

Inference only — no weights are updated. For every runs/<name>/ with a checkpoint:
reload it, score inner-val and test once each (eval mode: no modality dropout, no
augmentation, same preprocessing/slice selection as the normal evaluation), and write

  runs/<name>/features_val.npz     feat_{enc} (N,384), evidence_{enc}, subject_id, y, age, grade, site ...
  runs/<name>/features_test.npz
  runs/<name>/head_weights.npz     W_{enc} (2,384), b_{enc} (2,), row_order = [e1, e0]

Before anything is written, the head is recomputed by hand from (features, W, b)
for every subject and checked against the model's own evidence — that is the
row-order check (a swapped W1/W0 flips every sign downstream). When the run has a
predictions.csv, the fused alpha/beta of the first subject is also rebuilt from
the dumped evidence + volume scaling and compared with it.

    uv run python scripts/dump_features.py                       # the 12 shift runs + rand0_s0
    uv run python scripts/dump_features.py --runs smoke_foldA_s0 --include-all

After this, notebooks/05_shift_decomposition.ipynb needs no model and no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openidh_model.data.dataset import GliomaDataset  # noqa: E402
from openidh_model.eval.features import (  # noqa: E402
    ENCODERS, collect_features, head_weights, verify_heads, write_features, write_head_weights,
)
from openidh_model.models.fusion import volume_scale  # noqa: E402
from openidh_model.models.openidh import OpenIDH  # noqa: E402
from openidh_model.train.loop import make_loader  # noqa: E402
from openidh_model.utils.config import load_config  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402

# The 12 shift runs (4 units x 3 seeds) plus one in-distribution negative control.
SHIFT_RUNS = [f"{u}_s{s}" for u in ("foldA", "foldB", "field", "vendor") for s in range(3)]
CONTROL_RUNS = ["rand0_s0"]


def _dataset(split_csv, role, res, paths, cfg):
    ds = GliomaDataset(split_csv, paths, role=role, fold_id=res["fold_id"],
                       age_mean=res["age_mean"], age_std=res["age_std"])
    if cfg["data"].get("subset_n"):  # mirror run_fold.py so a smoke checkpoint stays smoke-sized
        from openidh_model.train.loop import _balanced_subset
        n = cfg["data"]["subset_n"]
        ds.rows = _balanced_subset(ds.rows, n if role == "test" else max(4, n // 2))
    return ds


def _check_against_predictions(run_dir: Path, feats: dict, role: str, cfg, tol=1e-3) -> str:
    """Rebuild the first subject's fused alpha/beta from the dumped raw evidence
    (image heads x volume scale, + tabular from the csv, +1) and compare."""
    pf = run_dir / "predictions.csv"
    if not pf.is_file():
        return "predictions.csv absent — fused check skipped"
    import pandas as pd
    p = pd.read_csv(pf, dtype={"active_mask": str, "subject_id": str})
    p = p[(p.role == role) & (p.active_mask == "111111")]
    sid = feats["subject_id"][0]
    row = p[p.subject_id == sid]
    if len(row) != 1:
        return f"{sid}: not found in predictions.csv (or partial mask) — fused check skipped"
    row = row.iloc[0]
    s = float(volume_scale(torch.tensor([feats["volume"][0]]), cfg["fusion"]["v_ref"],
                           cfg["fusion"]["gamma"])[0])
    e1 = sum(float(feats[f"evidence_{e}"][0, 0]) for e in ENCODERS) * s + float(row["e1_tabular"])
    e0 = sum(float(feats[f"evidence_{e}"][0, 1]) for e in ENCODERS) * s + float(row["e0_tabular"])
    da, db = abs(e1 + 1 - row["alpha"]), abs(e0 + 1 - row["beta"])
    ok = max(da, db) < tol * max(1.0, float(row["alpha"]), float(row["beta"]))
    msg = (f"{sid}: alpha {e1 + 1:.4f} vs csv {row['alpha']:.4f}, "
           f"beta {e0 + 1:.4f} vs csv {row['beta']:.4f}")
    if not ok:
        raise AssertionError("fused evidence does not match predictions.csv — " + msg)
    return msg + " (match)"


def dump_run(run_dir: Path, paths, device: str, log=print) -> dict | None:
    ckpt, res_f = run_dir / "model_best.pt", run_dir / "result.json"
    if not (ckpt.is_file() and res_f.is_file()):
        return None
    res = json.loads(res_f.read_text())
    cfg = load_config(paths.root / "configs", res.get("train_yaml", "train.yaml"))
    cfg["train"]["device"] = device

    model = OpenIDH(backbone=cfg["model"]["backbone"], weights_dir=paths.weights_dir)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    weights = head_weights(model)
    write_head_weights(run_dir / "head_weights.npz", weights)

    split_csv = paths.splits_dir / res["split_file"]
    out = {"run": run_dir.name, "split_file": res["split_file"], "fold_id": res["fold_id"],
           "seed": res["seed"], "checks": {}}
    for role in ("val", "test"):
        t0 = time.perf_counter()
        ds = _dataset(split_csv, role, res, paths, cfg)
        feats = collect_features(model, make_loader(ds, cfg, shuffle=False), cfg, device, paths)
        err = verify_heads(feats, weights)            # row-order / hand-computation check
        fused = _check_against_predictions(run_dir, feats, role, cfg)
        write_features(run_dir / f"features_{role}.npz", feats)
        out["checks"][role] = {"head_max_abs_err": err, "fused": fused}
        out[f"n_{role}"] = int(len(feats["y"]))
        log(f"  {role:4s} n={len(feats['y']):4d}  head recompute max|err|={max(err.values()):.2e}  "
            f"{fused}  ({time.perf_counter() - t0:.0f}s)")
    (run_dir / "features_meta.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--runs", nargs="*", default=None,
                    help=f"run names (default: {' '.join(SHIFT_RUNS + CONTROL_RUNS)})")
    ap.add_argument("--paths-file", default=None, help="defaults to $OPENIDH_PATHS")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--include-all", action="store_true",
                    help="allow smoke/probe checkpoints (default: train.yaml runs only)")
    a = ap.parse_args()

    paths = load_paths(a.paths_file)
    runs_dir = Path(a.runs_dir)
    names = a.runs or SHIFT_RUNS + CONTROL_RUNS

    done, skipped = [], []
    for n in names:
        d = runs_dir / n
        rj = d / "result.json"
        if not a.include_all and rj.is_file() and json.loads(rj.read_text()).get("train_yaml") != "train.yaml":
            skipped.append(f"{n} (not a train.yaml run; --include-all)"); continue
        if not (d / "model_best.pt").is_file():
            skipped.append(f"{n} (no checkpoint)"); continue
        print(f"[{len(done) + 1}/{len(names)}] {n}", flush=True)
        r = dump_run(d, paths, a.device)
        if r is None:
            skipped.append(f"{n} (no result.json)"); continue
        done.append(n)

    print(f"\ndumped: {done}")
    if skipped:
        print(f"skipped: {skipped}")
    if not done:
        raise SystemExit("nothing dumped — checkpoints are gitignored; run this where the sweep ran.")


if __name__ == "__main__":
    main()
