"""Apply post-hoc temperature scaling to finished runs — no retraining (spec §10).

For every runs/<name>/ that still has model_best.pt: reload the checkpoint, run
inference once on the inner-val split and once on test, fit T on inner-val, apply
it to test, and write runs/<name>/calibration.json with the raw and calibrated
metrics side by side. Prints a markdown table of ECE before/after per fold-unit.

    uv run python scripts/calibrate_runs.py                    # every run with a checkpoint
    uv run python scripts/calibrate_runs.py --runs foldB_s0    # just one

Inference only — the weights are never updated. AUC/AUPRC are asserted unchanged
(temperature is monotone); if that assertion fires, something else moved.
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
from openidh_model.models.openidh import OpenIDH  # noqa: E402
from openidh_model.train.calibrate import (  # noqa: E402
    calibrated_metrics, check_ranking_invariant, fit_temperature,
)
from openidh_model.train.loop import evaluate, make_loader  # noqa: E402
from openidh_model.utils.config import load_config  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402

METRICS = ("ece", "brier", "nll", "auc", "auprc")


def _alpha_beta(records):
    """evaluate() hands back S and p; recover the fused Beta from them."""
    S, p = np.asarray(records["S"]), np.asarray(records["p"])
    return p * S, S - p * S


def _infer(model, split_csv, role, res, paths, cfg, device):
    ds = GliomaDataset(split_csv, paths, role=role, fold_id=res["fold_id"],
                       age_mean=res["age_mean"], age_std=res["age_std"])
    if cfg["data"].get("subset_n"):  # mirror run_fold.py, so a smoke checkpoint stays smoke-sized
        from openidh_model.train.loop import _balanced_subset
        n = cfg["data"]["subset_n"]
        ds.rows = _balanced_subset(ds.rows, n if role == "test" else max(4, n // 2))
    _, rec = evaluate(model, make_loader(ds, cfg, shuffle=False), cfg, device)
    a, b = _alpha_beta(rec)
    return a, b, np.asarray(rec["y"]), len(ds)


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
    va, vb, vy, n_val = _infer(model, split_csv, "val", res, paths, cfg, device)
    ta, tb, ty, n_test = _infer(model, split_csv, "test", res, paths, cfg, device)

    T = fit_temperature(va, vb, vy)
    raw, cal = compute_metrics(ta, tb, ty), calibrated_metrics(ta, tb, ty, T)
    deltas = check_ranking_invariant(ta, tb, ty, T)

    out = {
        "run": run_dir.name, "split_file": res["split_file"], "fold_id": res["fold_id"],
        "seed": res["seed"], "temperature": T, "n_val": n_val, "n_test": n_test,
        "val_base_rate": float(np.mean(vy)), "test_base_rate": float(np.mean(ty)),
        "test_raw": raw, "test_calibrated": cal,
        "ranking_delta": deltas,
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

    out = ["| ユニット | 条件 | T (平均) | ECE 生 | ECE 較正後 | 改善 | Brier 生 → 較正後 | NLL 生 → 較正後 |",
           "|---|---|---|---|---|---|---|---|"]
    for lab in order:
        kind, rs = units[lab]
        er, ec = mean(rs, "test_raw", "ece"), mean(rs, "test_calibrated", "ece")
        br, bc = mean(rs, "test_raw", "brier"), mean(rs, "test_calibrated", "brier")
        nr, nc = mean(rs, "test_raw", "nll"), mean(rs, "test_calibrated", "nll")
        T = float(np.mean([x["temperature"] for x in rs]))
        out.append(f"| {lab} | {'分布外' if kind == 'OOD' else '分布内'} | {T:.2f} | "
                   f"{er:.3f} | {ec:.3f} | {(er - ec) / er * 100:+.0f}% | "
                   f"{br:.3f} → {bc:.3f} | {nr:.3f} → {nc:.3f} |")

    ood = [r for r in rows if _unit(r)[1] == "OOD"]
    ind = [r for r in rows if _unit(r)[1] == "in-dist"]
    out += ["", "| グループ | ECE 生 | ECE 較正後 | 改善 |", "|---|---|---|---|"]
    for nm, rs in (("分布外 (OOD)", ood), ("分布内", ind)):
        if not rs:
            continue
        er, ec = mean(rs, "test_raw", "ece"), mean(rs, "test_calibrated", "ece")
        out.append(f"| {nm} | {er:.3f} | {ec:.3f} | {(er - ec) / er * 100:+.0f}% |")

    d = max(abs(r["ranking_delta"].get(k, 0.0) or 0.0) for r in rows for k in ("auc", "auprc"))
    out += ["", f"検算: AUC / AUPRC の最大変化量 {d:.2e}（temperature scaling は単調変換なので 0 のはず）"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--runs", nargs="*", default=None, help="run names (default: all with a checkpoint)")
    ap.add_argument("--paths-file", default=None, help="defaults to $OPENIDH_PATHS")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    paths = load_paths(a.paths_file)
    runs_dir = Path(a.runs_dir)
    names = a.runs or sorted(d.name for d in runs_dir.iterdir()
                             if (d / "model_best.pt").is_file() and (d / "result.json").is_file())
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
              f"ECE {r['test_raw']['ece']:.3f} -> {r['test_calibrated']['ece']:.3f}", flush=True)

    if skipped:
        print(f"\nskipped (no checkpoint): {skipped}")
    rows = [r for r in rows if r["split_file"] in SPLIT_KIND]
    if rows:
        print("\n" + markdown_tables(rows))


if __name__ == "__main__":
    main()
