"""Run one (split, fold, seed): train on inner-train, select on inner-val,
evaluate the outer test exactly once, and save artifacts to output_dir (spec §10).

Logic lives here; notebooks/03_train.ipynb just calls run() (spec §11.5).
Outputs (model .pt, json) go under runs/ which is git-ignored.

CLI:
  uv run python scripts/run_fold.py --split-file splits_loso_foldA.csv \
      --seed 0 --output-dir runs/foldA_s0 [--train-yaml train.yaml] [--fold-id 0] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openidh_model.data.dataset import GliomaDataset  # noqa: E402
from openidh_model.eval.metrics import compute_metrics  # noqa: E402
from openidh_model.train.loop import evaluate, train_fold  # noqa: E402
from openidh_model.train.monitor import summarize  # noqa: E402
from openidh_model.utils.config import load_config  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402


def _subset(rows, n):
    from openidh_model.train.loop import _balanced_subset
    return _balanced_subset(rows, n)


def run(split_file, seed=0, output_dir="runs/default", train_yaml="train.yaml",
        fold_id=None, device=None, paths_file="configs/paths.local.yaml", log=print):
    paths = load_paths(paths_file)
    cfg = load_config(paths.root / "configs", train_yaml)
    if device:
        cfg["train"]["device"] = device
    dev = cfg["train"]["device"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log(f"=== fold: {split_file} fold_id={fold_id} seed={seed} device={dev} ===")
    res = train_fold(paths, cfg, split_file, fold_id=fold_id, seed=seed, log=log)

    # restore best (inner-val NLL) weights, then evaluate the outer test ONCE.
    model = res["model"]
    if "state" in res["best"]:
        model.load_state_dict(res["best"]["state"])
    model.to(dev)

    test_ds = GliomaDataset(paths.splits_dir / split_file, paths, role="test",
                            fold_id=fold_id, age_mean=res["age_mean"], age_std=res["age_std"])
    if cfg["data"].get("subset_n"):  # keep the local smoke fast
        test_ds.rows = _subset(test_ds.rows, cfg["data"]["subset_n"])
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"])
    test_metrics, test_rec = evaluate(model, test_loader, cfg, dev)
    mon = summarize(test_rec)
    log(f"TEST metrics: { {k: round(v,4) for k,v in test_metrics.items()} }")

    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model_best.pt")
    (out / "history.json").write_text(json.dumps(res["history"], indent=2))
    (out / "result.json").write_text(json.dumps({
        "split_file": split_file, "fold_id": fold_id, "seed": seed,
        "train_yaml": train_yaml, "best_epoch": res["best"]["epoch"],
        "best_val_nll": res["best"]["nll"],
        "age_mean": res["age_mean"], "age_std": res["age_std"],
        "n_train": len(res["train_ds"]), "n_val": len(res["val_ds"]), "n_test": len(test_ds),
        "test_metrics": test_metrics, "test_monitor": mon,
    }, indent=2, default=float))
    log(f"saved -> {out}")
    return {"test_metrics": test_metrics, "monitor": mon, "output_dir": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--fold-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="runs/default")
    ap.add_argument("--train-yaml", default="train.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--paths-file", default="configs/paths.local.yaml")
    a = ap.parse_args()
    run(a.split_file, a.seed, a.output_dir, a.train_yaml, a.fold_id, a.device, a.paths_file)


if __name__ == "__main__":
    main()
