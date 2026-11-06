"""External validation: score trained checkpoints on EGD and Japan — inference only.

For every run (default: the 6 LOSO runs foldA_s0-2, foldB_s0-2), reload
runs/<name>/model_best.pt and score each cohort split written by
build_external_splits.py. No weights are updated. Each run uses its own
training age mean/SD (result.json) and its own inner-val temperature
(calibration.json), so the input is exactly what that run saw at test time.

Cases with no age or sex are scored without the tabular head (train.loop._forward);
`active_mask` in predictions.csv records which heads were used.

Outputs (under --out-dir, outside the repo — the per-case file carries labels):
  predictions.csv  one row per (run, cohort, subject), same columns as runs/*/predictions.csv
                   (`role` = cohort name)
  summary.json     metrics per run x cohort x subset, raw and temperature-scaled,
                   plus mean/sd over seeds per training unit (LOSO-A / LOSO-B)
  summary.md       the same as tables, with the run's own LOSO test result for reference

Subsets: `all` (every case in the split) and `all4` (T1, T2, FLAIR and T1c all present).

    uv run python scripts/eval_external.py                     # 6 LOSO runs x EGD, Japan
    uv run python scripts/eval_external.py --runs foldA_s0 --cohorts EGD
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

from openidh_model.data.dataset import GliomaDataset  # noqa: E402
from openidh_model.data.metadata import EXTERNAL_COHORTS  # noqa: E402
from openidh_model.eval.metrics import compute_metrics  # noqa: E402
from openidh_model.eval.predict import collect_predictions, write_predictions  # noqa: E402
from openidh_model.models.openidh import OpenIDH  # noqa: E402
from openidh_model.train.calibrate import apply_temperature  # noqa: E402
from openidh_model.train.loop import make_loader  # noqa: E402
from openidh_model.utils.config import load_config  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402

LOSO_RUNS = [f"{u}_s{s}" for u in ("foldA", "foldB") for s in range(3)]
UNIT = {"foldA": "LOSO-A", "foldB": "LOSO-B"}
METRICS = ["auc", "auprc", "ece", "nll", "brier"]


def _metrics(p: pd.DataFrame, T: float | None) -> dict:
    a, b, y = p["alpha"].to_numpy(), p["beta"].to_numpy(), p["y_true"].to_numpy()
    m = compute_metrics(a, b, y)
    out = {k: m[k] for k in METRICS}
    out.update(n=int(len(p)), n_mut=int(y.sum()), base_rate=float(y.mean()),
               mean_p=float((a / (a + b)).mean()),
               n_no_tabular=int((p["active_mask"].str[-1] == "0").sum()))
    if T is not None:
        ta, tb = apply_temperature(a, b, T)
        mt = compute_metrics(ta, tb, y)
        out["temp"] = {k: mt[k] for k in ("ece", "nll", "brier")}
        out["temp"]["mean_p"] = float((ta / (ta + tb)).mean())
    return out


def score_run(run_dir: Path, splits: dict, paths, device: str, log=print) -> list[dict]:
    res = json.loads((run_dir / "result.json").read_text())
    cfg = load_config(paths.root / "configs", res.get("train_yaml", "train.yaml"))
    cfg["train"]["device"] = device

    model = OpenIDH(backbone=cfg["model"]["backbone"], weights_dir=paths.weights_dir)
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device))
    model.to(device).eval()

    rows = []
    for cohort, split_csv in splits.items():
        t0 = time.perf_counter()
        ds = GliomaDataset(split_csv, paths, role="test",
                           age_mean=res["age_mean"], age_std=res["age_std"])
        if ds.missing:
            log(f"  !! {cohort}: {len(ds.missing)} subjects have no volumes/ dir: {ds.missing[:10]}")
        r = collect_predictions(model, make_loader(ds, cfg, shuffle=False), cfg, device,
                                role=cohort, run=run_dir.name)
        rows += r
        log(f"  {cohort:6s} n={len(r):4d}  ({time.perf_counter() - t0:.0f}s)")
    return rows


def summarize(pred: pd.DataFrame, splits: dict, runs_dir: Path) -> dict:
    all4 = {c: set(pd.read_csv(f, dtype={"subject_id": str}).query("all4 == 1").subject_id)
            for c, f in splits.items()}
    per_run: dict = {}
    for run, g in pred.groupby("run"):
        rd = runs_dir / run
        cal = json.loads((rd / "calibration.json").read_text()) if (rd / "calibration.json").is_file() else {}
        T = cal.get("temperature")
        own = json.loads((rd / "result.json").read_text())["test_metrics"]
        per_run[run] = {"temperature": T, "own_test": {k: own[k] for k in METRICS}}
        for cohort, gc in g.groupby("role"):
            per_run[run][cohort] = {
                "all": _metrics(gc, T),
                "all4": _metrics(gc[gc.subject_id.isin(all4[cohort])], T),
            }

    by_unit: dict = {}
    for prefix, unit in UNIT.items():
        runs = [r for r in per_run if r.startswith(prefix + "_")]
        if not runs:
            continue
        by_unit[unit] = {"runs": runs}
        for cohort in splits:
            for sub in ("all", "all4"):
                cells = [per_run[r][cohort][sub] for r in runs if cohort in per_run[r]]
                if not cells:
                    continue
                agg = {}
                for k in METRICS:
                    v = np.array([c[k] for c in cells], dtype=float)
                    agg[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
                if all("temp" in c for c in cells):
                    v = np.array([c["temp"]["ece"] for c in cells])
                    agg["ece_temp"] = {"mean": float(v.mean()),
                                       "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
                agg.update(n=cells[0]["n"], n_mut=cells[0]["n_mut"],
                           base_rate=cells[0]["base_rate"],
                           n_no_tabular=cells[0]["n_no_tabular"])
                by_unit[unit].setdefault(cohort, {})[sub] = agg
        own = np.array([[per_run[r]["own_test"][k] for k in METRICS] for r in runs])
        by_unit[unit]["own_test"] = {k: {"mean": float(own[:, i].mean()),
                                         "sd": float(own[:, i].std(ddof=1)) if len(runs) > 1 else 0.0}
                                     for i, k in enumerate(METRICS)}
    return {"per_run": per_run, "by_unit": by_unit}


def to_markdown(s: dict, cohorts: list[str]) -> str:
    f = lambda d: f"{d['mean']:.3f} ± {d['sd']:.3f}"  # noqa: E731
    out = ["# External validation — LOSO checkpoints on EGD / Japan", "",
           "Mean ± SD over 3 seeds. ECE (T) = after each run's inner-val temperature. "
           "`own test` = the run's own LOSO test set, for reference.", ""]
    for unit, u in s["by_unit"].items():
        out += [f"## {unit} ({', '.join(u['runs'])})", "",
                "| cohort | subset | n (Mut) | rate | no age/sex | AUC | AUPRC | ECE | ECE (T) | NLL | Brier |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        o = u["own_test"]
        out.append(f"| own test | — | — | — | — | {f(o['auc'])} | {f(o['auprc'])} | {f(o['ece'])} "
                   f"| — | {f(o['nll'])} | {f(o['brier'])} |")
        for c in cohorts:
            for sub, a in u.get(c, {}).items():
                out.append(
                    f"| {c} | {sub} | {a['n']} ({a['n_mut']}) | {a['base_rate']:.1%} | {a['n_no_tabular']} "
                    f"| {f(a['auc'])} | {f(a['auprc'])} | {f(a['ece'])} "
                    f"| {f(a['ece_temp']) if 'ece_temp' in a else '—'} | {f(a['nll'])} | {f(a['brier'])} |")
        out.append("")
    out += ["## Per run", "", "| run | cohort | subset | n | AUC | ECE | ECE (T) | mean p̂ | rate |",
            "|---|---|---|---|---|---|---|---|---|"]
    for run, r in s["per_run"].items():
        for c in cohorts:
            for sub, a in r.get(c, {}).items():
                et = f"{a['temp']['ece']:.3f}" if "temp" in a else "—"
                out.append(f"| {run} | {c} | {sub} | {a['n']} | {a['auc']:.3f} | {a['ece']:.3f} | {et} "
                           f"| {a['mean_p']:.3f} | {a['base_rate']:.3f} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=LOSO_RUNS)
    ap.add_argument("--cohorts", nargs="*", default=list(EXTERNAL_COHORTS))
    ap.add_argument("--runs-dir", default=None, help="default: paths output_dir (runs/)")
    ap.add_argument("--splits-dir", default=None, help="default: <external_metadata_dir>/splits")
    ap.add_argument("--out-dir", default=None, help="default: <external_metadata_dir>/../external_eval")
    ap.add_argument("--paths-file", default=None, help="defaults to $OPENIDH_PATHS")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = load_paths(args.paths_file)
    if paths.external_metadata_dir is None:
        sys.exit("external_metadata_dir is not set in the paths YAML")
    runs_dir = Path(args.runs_dir or paths.output_dir)
    splits_dir = Path(args.splits_dir or paths.external_metadata_dir / "splits")
    out_dir = Path(args.out_dir or paths.external_metadata_dir.parent / "external_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {c: splits_dir / f"splits_ext_{c.lower()}.csv" for c in args.cohorts}
    missing = [str(f) for f in splits.values() if not f.is_file()]
    if missing:
        sys.exit(f"split CSV not found — run scripts/build_external_splits.py first: {missing}")
    no_ckpt = [r for r in args.runs if not (runs_dir / r / "model_best.pt").is_file()]
    if no_ckpt:
        sys.exit(f"no model_best.pt for: {no_ckpt} (under {runs_dir})")

    rows = []
    for run in args.runs:
        print(f"== {run}")
        rows += score_run(runs_dir / run, splits, paths, args.device)
    write_predictions(rows, out_dir / "predictions.csv")

    pred = pd.read_csv(out_dir / "predictions.csv", dtype={"subject_id": str, "active_mask": str})
    s = summarize(pred, splits, runs_dir)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
    (out_dir / "summary.md").write_text(to_markdown(s, args.cohorts))
    print(to_markdown(s, args.cohorts))
    print(f"-> {out_dir}/{{predictions.csv,summary.json,summary.md}}")


if __name__ == "__main__":
    main()
