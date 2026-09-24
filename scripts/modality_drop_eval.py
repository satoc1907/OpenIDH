"""Score every run as if some sequences had never been acquired (inference only).

The head-level ablation (scripts/head_ablation.py) answers "what if we ignored
what the T2/FLAIR heads say" using arithmetic on predictions.csv. It cannot
answer "what if those sequences were missing", because the unified trunk still
receives their channels. This does: the named modalities are zeroed before the
forward pass — so the unified trunk sees the absence too — and their heads leave
the fusion, which is exactly the missing-sequence path the model was trained
with (modality dropout p=0.2, spec §3.5 / §7).

For each run and each drop-set, inner-val and test are scored once and written as

  runs/<run>/predictions_drop-<tag>.csv     same schema as predictions.csv

so the comparison afterwards needs no GPU. scripts/head_ablation.py picks these
files up automatically and puts them in the same table as the head-level rows.

    uv run python scripts/modality_drop_eval.py                     # 13 runs x 2 drop-sets
    uv run python scripts/modality_drop_eval.py --runs vendor_s0 --drops T2+FLAIR

Inference only — no weights are updated, nothing is retrained. The baseline
(nothing dropped) is re-scored for the first run only, as a check that this path
reproduces the existing predictions.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from dump_features import CONTROL_RUNS, SHIFT_RUNS, _dataset  # noqa: E402
from openidh_model.eval.metrics import compute_metrics  # noqa: E402
from openidh_model.eval.predict import collect_predictions, write_predictions  # noqa: E402
from openidh_model.models.openidh import OpenIDH  # noqa: E402
from openidh_model.train.loop import make_loader  # noqa: E402
from openidh_model.utils.config import load_config  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402

# tag -> modalities treated as never acquired. Ordered cheapest-question-first.
DROPS = {
    "T2+FLAIR": ["T2", "FLAIR"],            # the proposal: the two shift-damaged sequences
    "T1+T2+FLAIR": ["T1", "T2", "FLAIR"],   # all three weak single-sequence encoders
}


def tag_path(run_dir: Path, tag: str) -> Path:
    return run_dir / f"predictions_drop-{tag}.csv"


def _score(rows: list[dict]) -> dict:
    g = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    test = [r for r in rows if r["role"] == "test"]
    gt = lambda k: np.array([r[k] for r in test], dtype=float)  # noqa: E731
    return compute_metrics(gt("alpha"), gt("beta"), gt("y_true")) | {"n_test": len(test), "n": len(rows)}


def eval_run(run_dir: Path, paths, device: str, drops: dict, baseline_check: bool, log=print) -> dict | None:
    ckpt, res_f = run_dir / "model_best.pt", run_dir / "result.json"
    if not (ckpt.is_file() and res_f.is_file()):
        return None
    res = json.loads(res_f.read_text())
    cfg = load_config(paths.root / "configs", res.get("train_yaml", "train.yaml"))
    cfg["train"]["device"] = device

    model = OpenIDH(backbone=cfg["model"]["backbone"], weights_dir=paths.weights_dir)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    split_csv = paths.splits_dir / res["split_file"]
    out = {"run": run_dir.name, "split_file": res["split_file"], "seed": res["seed"], "drops": {}}

    def _pass(drop_list, tag):
        rows = []
        for role in ("val", "test"):
            ds = _dataset(split_csv, role, res, paths, cfg)
            rows += collect_predictions(model, make_loader(ds, cfg, shuffle=False), cfg, device,
                                        role, run_dir.name, drop=drop_list)
        return rows

    if baseline_check:  # does this path reproduce the stored predictions.csv?
        t0 = time.perf_counter()
        rows = _pass(None, "none")
        ref_f = run_dir / "predictions.csv"
        msg = "predictions.csv absent — baseline check skipped"
        if ref_f.is_file():
            ref = pd.read_csv(ref_f, dtype={"subject_id": str, "active_mask": str})
            new = pd.DataFrame(rows)
            m = ref.merge(new, on=["role", "subject_id"], suffixes=("_ref", "_new"))
            err = max(float((m["alpha_ref"] - m["alpha_new"]).abs().max()),
                      float((m["beta_ref"] - m["beta_new"]).abs().max()))
            assert len(m) == len(ref) and err < 1e-2, f"baseline mismatch: n={len(m)}/{len(ref)} err={err:.2e}"
            msg = f"max |Δα|,|Δβ| = {err:.2e} over {len(m)} subjects (match)"
        out["baseline_check"] = msg
        log(f"  baseline: {msg}  ({time.perf_counter() - t0:.0f}s)")

    for tag, mods in drops.items():
        t0 = time.perf_counter()
        rows = _pass(mods, tag)
        write_predictions(rows, tag_path(run_dir, tag))
        s = _score(rows)
        # every row must record the drop in its mask (T1,T2,FLAIR,T1c,unified,tabular order)
        masks = {r["active_mask"] for r in rows}
        out["drops"][tag] = {"dropped": mods, "active_masks": sorted(masks), **s}
        log(f"  drop {tag:12s} test n={s['n_test']:4d} auc={s['auc']:.3f} ece={s['ece']:.3f} "
            f"nll={s['nll']:.3f} medS={s['median_S']:.1f} mask={sorted(masks)} "
            f"({time.perf_counter() - t0:.0f}s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--drops", nargs="*", default=None, help=f"tags from {list(DROPS)}")
    ap.add_argument("--paths-file", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--include-all", action="store_true")
    ap.add_argument("--no-baseline-check", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    paths = load_paths(a.paths_file)
    runs_dir = Path(a.runs_dir)
    names = a.runs or SHIFT_RUNS + CONTROL_RUNS
    drops = {t: DROPS[t] for t in (a.drops or DROPS)}

    done, skipped, results = [], [], []
    for i, n in enumerate(names):
        d = runs_dir / n
        rj = d / "result.json"
        if not a.include_all and rj.is_file() and json.loads(rj.read_text()).get("train_yaml") != "train.yaml":
            skipped.append(f"{n} (not a train.yaml run)"); continue
        if not (d / "model_best.pt").is_file():
            skipped.append(f"{n} (no checkpoint)"); continue
        print(f"[{i + 1}/{len(names)}] {n}", flush=True)
        r = eval_run(d, paths, a.device, drops, baseline_check=(not a.no_baseline_check and not done))
        if r is None:
            skipped.append(f"{n} (no result.json)"); continue
        results.append(r); done.append(n)

    print(f"\ndumped: {done}")
    if skipped:
        print(f"skipped: {skipped}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(results, indent=1, default=float))
        print(f"wrote {a.json}")
    if not done:
        raise SystemExit("nothing scored — checkpoints are gitignored; run this where the sweep ran.")


if __name__ == "__main__":
    main()
