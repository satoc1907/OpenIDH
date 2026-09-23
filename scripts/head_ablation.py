"""What if a head were simply switched off? (head-level modality ablation)

Fusion is a sum — alpha = sum_h e1_h + 1, beta = sum_h e0_h + 1 (spec §5) — and
predictions.csv already stores the post-scaling, post-masking e1/e0 of every
head. So "declare T2 and FLAIR unknown" is exact arithmetic on a file we
already have: drop those columns from the sum. No inference, no GPU, no
retraining. The reconstruction of the untouched sum is asserted against the
stored alpha/beta before anything is reported.

    uv run python scripts/head_ablation.py
    uv run python scripts/head_ablation.py --json results/head_ablation.json

CAVEAT, stated in every output: this switches off the HEADS. The unified trunk
still receives the T2/FLAIR channels, so this is "ignore what those heads say",
not "the sequences were never acquired". The latter needs re-inference with the
channels zeroed (the modality-dropout path the model was trained with, p=0.2).
That is the follow-up this script is meant to justify or rule out.

Two ways to read a subset's test score, and they answer different questions:
  oracle     the best subset ON TEST — an upper bound, not deployable
  val-picked the subset chosen on inner-val (by NLL), then applied to test —
             what a deployment rule could actually do
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from aggregate_results import SPLIT_KIND  # noqa: E402
from openidh_model.eval.metrics import compute_metrics  # noqa: E402
from openidh_model.train.monitor import CLASSIFIERS  # noqa: E402

# Subsets worth asking about. "keep" lists the heads that stay ON.
IMAGE = ["T1", "T2", "FLAIR", "T1c"]
SUBSETS = {
    "all (baseline)": CLASSIFIERS,
    "-T2,FLAIR": [h for h in CLASSIFIERS if h not in ("T2", "FLAIR")],
    "-T2": [h for h in CLASSIFIERS if h != "T2"],
    "-FLAIR": [h for h in CLASSIFIERS if h != "FLAIR"],
    "-T1,T2,FLAIR (weak singles)": [h for h in CLASSIFIERS if h not in ("T1", "T2", "FLAIR")],
    "T1c+unified+tab": ["T1c", "unified", "tabular"],
    "T1c+tab": ["T1c", "tabular"],
    "unified+tab": ["unified", "tabular"],
    "-unified": [h for h in CLASSIFIERS if h != "unified"],
    "-tabular": [h for h in CLASSIFIERS if h != "tabular"],
}


def fuse_subset(df: pd.DataFrame, keep: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """alpha, beta from the kept heads only (the fusion of spec §5, restated)."""
    e1 = sum(df[f"e1_{h}"].to_numpy(float) for h in keep)
    e0 = sum(df[f"e0_{h}"].to_numpy(float) for h in keep)
    return e1 + 1.0, e0 + 1.0


def _uncertainty_ok(alpha, beta, y) -> float:
    """median S among correct minus median S among wrong (spec §6.2: must be >= 0)."""
    S = alpha + beta
    p = alpha / S
    ok = (p > 0.5).astype(int) == np.asarray(y, int)
    if ok.all() or (~ok).any() is False:
        return float("nan")
    return float(np.median(S[ok]) - np.median(S[~ok]))


def score(df: pd.DataFrame, keep: list[str]) -> dict:
    a, b = fuse_subset(df, keep)
    y = df["y_true"].to_numpy(float)
    m = compute_metrics(a, b, y)
    m["S_gap"] = _uncertainty_ok(a, b, y)
    return m


def run_table(runs_dir: Path, runs: list[str]) -> pd.DataFrame:
    rows = []
    for run in runs:
        pf = runs_dir / run / "predictions.csv"
        rj = runs_dir / run / "result.json"
        if not (pf.is_file() and rj.is_file()):
            continue
        res = json.loads(rj.read_text())
        if res.get("train_yaml") != "train.yaml" or res["split_file"] not in SPLIT_KIND:
            continue
        label, kind = SPLIT_KIND[res["split_file"]]
        if res.get("fold_id") not in (None, ""):
            label = f"{label} f{res['fold_id']}"
        df = pd.read_csv(pf)
        # the sum must reproduce the stored fusion, or the columns mean something else
        a, b = fuse_subset(df, CLASSIFIERS)
        err = max(np.abs(a - df["alpha"]).max(), np.abs(b - df["beta"]).max())
        assert err < 1e-3, f"{run}: fusion reconstruction off by {err:.2e}"
        for role in ("val", "test"):
            d = df[df.role == role]
            for name, keep in SUBSETS.items():
                rows.append({"run": run, "unit": label, "kind": kind, "seed": res["seed"],
                             "role": role, "subset": name, "n_keep": len(keep), "n": len(d),
                             **score(d, keep)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """mean over seeds per unit x subset, on test."""
    t = df[df.role == "test"]
    g = t.groupby(["unit", "kind", "subset"], observed=True)
    cols = ["auc", "auprc", "ece", "nll", "brier", "median_S", "S_gap"]
    out = pd.concat([g[cols].mean().round(4), g.size().rename("n_runs")], axis=1).reset_index()
    return out


def val_picked(df: pd.DataFrame, metric: str = "nll") -> pd.DataFrame:
    """Per run: the subset with the best inner-val `metric`, and its TEST score.

    This is the deployable reading — the choice never sees the test set. Reported
    next to the always-on baseline and the test-side oracle.
    """
    rows = []
    for run, g in df.groupby("run", observed=True):
        v, t = g[g.role == "val"], g[g.role == "test"]
        pick = v.loc[v[metric].idxmin(), "subset"]
        best_t = t.loc[t["nll"].idxmin(), "subset"]
        base = t[t.subset == "all (baseline)"].iloc[0]
        got = t[t.subset == pick].iloc[0]
        ora = t[t.subset == best_t].iloc[0]
        rows.append({"run": run, "unit": g["unit"].iloc[0], "kind": g["kind"].iloc[0],
                     "val_pick": pick, "test_oracle": best_t,
                     "auc_base": base["auc"], "auc_pick": got["auc"], "auc_oracle": ora["auc"],
                     "nll_base": base["nll"], "nll_pick": got["nll"], "nll_oracle": ora["nll"],
                     "ece_base": base["ece"], "ece_pick": got["ece"], "ece_oracle": ora["ece"]})
    return pd.DataFrame(rows)


def markdown(summ: pd.DataFrame, picked: pd.DataFrame) -> str:
    out = ["| ユニット | 条件 | 構成 | AUC | AUPRC | ECE | NLL | median S | S(正)−S(誤) |",
           "|---|---|---|---|---|---|---|---|---|"]
    order = sorted(summ["unit"].unique(), key=lambda u: (summ[summ.unit == u]["kind"].iloc[0] != "OOD", u))
    for u in order:
        s = summ[summ.unit == u]
        kind = "分布外" if s["kind"].iloc[0] == "OOD" else "分布内"
        base = s[s.subset == "all (baseline)"].iloc[0]
        for _, r in s.iterrows():
            mark = ""
            if r["subset"] != "all (baseline)":
                mark = f" ({r['auc'] - base['auc']:+.3f})"
            out.append(f"| {u} | {kind} | {r['subset']} | {r['auc']:.3f}{mark} | {r['auprc']:.3f} | "
                       f"{r['ece']:.3f} | {r['nll']:.3f} | {r['median_S']:.1f} | {r['S_gap']:+.1f} |")
    out += ["", "| ユニット | val が選ぶ構成 | AUC 全部→val選択→test最良 | NLL 全部→val選択→test最良 |",
            "|---|---|---|---|"]
    for u in order:
        p = picked[picked.unit == u]
        if not len(p):
            continue
        picks = "/".join(sorted(set(p["val_pick"])))
        out.append(f"| {u} | {picks} | {p['auc_base'].mean():.3f} → {p['auc_pick'].mean():.3f} → "
                   f"{p['auc_oracle'].mean():.3f} | {p['nll_base'].mean():.3f} → "
                   f"{p['nll_pick'].mean():.3f} → {p['nll_oracle'].mean():.3f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs_dir = Path(a.runs_dir)
    runs = a.runs or sorted(d.name for d in runs_dir.iterdir() if (d / "predictions.csv").is_file())

    df = run_table(runs_dir, runs)
    if not len(df):
        raise SystemExit("no production run with predictions.csv found")
    summ = summarize(df)
    picked = val_picked(df)
    print(f"{df['run'].nunique()} runs x {len(SUBSETS)} subsets x (val, test)\n")
    print(markdown(summ, picked))
    print("\n注意: これはヘッドを落としただけで、unified trunk には T2/FLAIR チャネルが入ったまま。")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"per_run": df.to_dict("records"), "summary": summ.to_dict("records"),
             "val_picked": picked.to_dict("records"),
             "subsets": {k: v for k, v in SUBSETS.items()}}, indent=1, default=float))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
