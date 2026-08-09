"""Post-hoc calibration over finished runs — no retraining (spec §10).

For every runs/<name>/ that still has model_best.pt: reload the checkpoint, score
the inner-val split and the test split once each, and write

  runs/<name>/predictions.csv   one row per subject (fused Beta + per-head evidence)
  runs/<name>/calibration.json  raw / temperature / prior-oracle metrics side by side

Two corrections are reported, because they answer different questions:
  temperature   T fitted on inner-val — the deployable one, and the one §10 asked for
  prior oracle  logit offset from the OBSERVED test prevalence — not deployable, but
                it bounds how much of the OOD miscalibration is pure base-rate shift

    uv run python scripts/calibrate_runs.py                    # every production run
    uv run python scripts/calibrate_runs.py --runs foldB_s0    # just one

Inference only — the weights are never updated. AUC/AUPRC are asserted unchanged
(both corrections are monotone); if that assertion fires, something else moved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from aggregate_results import SPLIT_KIND  # noqa: E402
from openidh_model.data.dataset import GliomaDataset  # noqa: E402
from openidh_model.eval.metrics import compute_metrics  # noqa: E402
from openidh_model.eval.predict import collect_predictions, write_predictions  # noqa: E402
from openidh_model.models.openidh import OpenIDH  # noqa: E402
from openidh_model.train.calibrate import (  # noqa: E402
    calibrated_metrics, check_ranking_invariant, fit_temperature, prior_offset,
)
from openidh_model.train.loop import make_loader  # noqa: E402
from openidh_model.utils.config import load_config  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402


def _infer(model, split_csv, role, res, paths, cfg, device, run):
    ds = GliomaDataset(split_csv, paths, role=role, fold_id=res["fold_id"],
                       age_mean=res["age_mean"], age_std=res["age_std"])
    if cfg["data"].get("subset_n"):  # mirror run_fold.py, so a smoke checkpoint stays smoke-sized
        from openidh_model.train.loop import _balanced_subset
        n = cfg["data"]["subset_n"]
        ds.rows = _balanced_subset(ds.rows, n if role == "test" else max(4, n // 2))
    rows = collect_predictions(model, make_loader(ds, cfg, shuffle=False), cfg, device, role, run)
    g = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    return g("alpha"), g("beta"), g("y_true"), len(ds), rows


def calibrate_run(run_dir: Path, paths, device: str) -> dict | None:
    ckpt = run_dir / "model_best.pt"
    res_f = run_dir / "result.json"
    if not (ckpt.is_file() and res_f.is_file()):
        return None
    res = json.loads(res_f.read_text())
    cfg = load_config(paths.root / "configs", res.get("train_yaml", "train.yaml"))
    cfg["train"]["device"] = device

    model = OpenIDH(backbone=cfg["model"]["backbone"], weights_dir=paths.weights_dir)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    split_csv = paths.splits_dir / res["split_file"]
    run = run_dir.name
    va, vb, vy, n_val, vrows = _infer(model, split_csv, "val", res, paths, cfg, device, run)
    ta, tb, ty, n_test, trows = _infer(model, split_csv, "test", res, paths, cfg, device, run)
    write_predictions(vrows + trows, run_dir / "predictions.csv")

    pi_val, pi_test = float(np.mean(vy)), float(np.mean(ty))
    T = fit_temperature(va, vb, vy)
    # Oracle: the offset uses the OBSERVED test prevalence, so it is an upper bound
    # on base-rate correction, not a deployable method (spec §10 diagnosed exactly
    # this shift on fold B, but temperature has no intercept to express it).
    offset = prior_offset(pi_val, pi_test)

    raw = compute_metrics(ta, tb, ty)
    cal = calibrated_metrics(ta, tb, ty, T=T)
    ora = calibrated_metrics(ta, tb, ty, offset=offset)
    deltas = check_ranking_invariant(ta, tb, ty, T=T)
    deltas_o = check_ranking_invariant(ta, tb, ty, offset=offset)

    out = {
        "run": run, "split_file": res["split_file"], "fold_id": res["fold_id"],
        "seed": res["seed"], "train_yaml": res.get("train_yaml", "train.yaml"),
        "temperature": T, "prior_offset": offset, "n_val": n_val, "n_test": n_test,
        "val_base_rate": pi_val, "test_base_rate": pi_test,
        "test_raw": raw, "test_calibrated": cal, "test_prior_oracle": ora,
        "ranking_delta": deltas, "ranking_delta_prior": deltas_o,
    }
    (run_dir / "calibration.json").write_text(json.dumps(out, indent=2))
    return out


def _unit(r) -> tuple[str, str]:
    label, kind = SPLIT_KIND[r["split_file"]]
    if r["fold_id"] not in (None, ""):
        label = f"{label} f{r['fold_id']}"
    return label, kind


def markdown_tables(rows: list[dict]) -> str:
    units, order = {}, []
    for r in rows:
        lab, kind = _unit(r)
        if lab not in units:
            units[lab] = (kind, []); order.append(lab)
        units[lab][1].append(r)
    order.sort(key=lambda l: (units[l][0] != "OOD", l))

    def mean(rs, sect, k):
        return float(np.mean([x[sect][k] for x in rs]))

    has_oracle = all("test_prior_oracle" in r for r in rows)
    pct = lambda a, b: f"{(a - b) / a * 100:+.0f}%" if a else "n/a"  # noqa: E731

    out = ["| ユニット | 条件 | T | prior offset | ECE 生 | ECE temp | ECE prior(oracle) | 生→oracle |",
           "|---|---|---|---|---|---|---|---|"]
    for lab in order:
        kind, rs = units[lab]
        er = mean(rs, "test_raw", "ece")
        ec = mean(rs, "test_calibrated", "ece")
        eo = mean(rs, "test_prior_oracle", "ece") if has_oracle else float("nan")
        T = float(np.mean([x["temperature"] for x in rs]))
        off = float(np.mean([x.get("prior_offset", float("nan")) for x in rs]))
        out.append(f"| {lab} | {'分布外' if kind == 'OOD' else '分布内'} | {T:.2f} | {off:+.2f} | "
                   f"{er:.3f} | {ec:.3f} | {eo:.3f} | {pct(er, eo)} |")

    ood = [r for r in rows if _unit(r)[1] == "OOD"]
    ind = [r for r in rows if _unit(r)[1] == "in-dist"]
    out += ["", "| グループ | ECE 生 | ECE temp | ECE prior(oracle) | Brier 生→oracle | NLL 生→oracle |",
            "|---|---|---|---|---|---|"]
    for nm, rs in (("分布外 (OOD)", ood), ("分布内", ind)):
        if not rs:
            continue
        er, ec = mean(rs, "test_raw", "ece"), mean(rs, "test_calibrated", "ece")
        eo = mean(rs, "test_prior_oracle", "ece") if has_oracle else float("nan")
        br, bo = mean(rs, "test_raw", "brier"), mean(rs, "test_prior_oracle", "brier")
        nr, no = mean(rs, "test_raw", "nll"), mean(rs, "test_prior_oracle", "nll")
        out.append(f"| {nm} | {er:.3f} | {ec:.3f} | {eo:.3f} | "
                   f"{br:.3f} → {bo:.3f} | {nr:.3f} → {no:.3f} |")

    keys = ["ranking_delta"] + (["ranking_delta_prior"] if has_oracle else [])
    d = max(abs(r[k].get(m, 0.0) or 0.0) for r in rows for k in keys for m in ("auc", "auprc"))
    out += ["", f"検算: AUC / AUPRC の最大変化量 {d:.2e}"
                "（temperature も prior offset も単調変換なので 0 のはず）"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--runs", nargs="*", default=None, help="run names (default: all with a checkpoint)")
    ap.add_argument("--paths-file", default=None, help="defaults to $OPENIDH_PATHS")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--include-all", action="store_true",
                    help="include smoke/probe runs (default: train.yaml runs only)")
    a = ap.parse_args()

    paths = load_paths(a.paths_file)
    runs_dir = Path(a.runs_dir)

    def _production(d: Path) -> bool:
        """Same gate as aggregate_results: a smoke checkpoint must not land in the table."""
        if a.include_all:
            return True
        rj = d / "result.json"
        return rj.is_file() and json.loads(rj.read_text()).get("train_yaml") == "train.yaml"

    names = a.runs or sorted(d.name for d in runs_dir.iterdir()
                             if (d / "model_best.pt").is_file() and (d / "result.json").is_file()
                             and _production(d))
    if not names:
        raise SystemExit(f"no run under {runs_dir}/ has both model_best.pt and result.json — "
                         "checkpoints are gitignored, so run this where the sweep ran.")

    rows, skipped = [], []
    for n in names:
        r = calibrate_run(runs_dir / n, paths, a.device)
        if r is None:
            skipped.append(n); continue
        rows.append(r)
        print(f"[{len(rows)}/{len(names)}] {n:14s} T={r['temperature']:.3f}  "
              f"ECE {r["test_raw"]["ece"]:.3f} -> temp {r["test_calibrated"]["ece"]:.3f} / prior {r["test_prior_oracle"]["ece"]:.3f}", flush=True)

    if skipped:
        print(f"\nskipped (no checkpoint): {skipped}")
    rows = [r for r in rows if r["split_file"] in SPLIT_KIND]
    if rows:
        print("\n" + markdown_tables(rows))


if __name__ == "__main__":
    main()
