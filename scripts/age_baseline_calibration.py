"""Is the age-only model calibrated where the image model is not?

The image model keeps its AUC under site shift but loses its calibration. The
age-only logistic regression was compared on AUC only (age_prior_eval.auc_table).
This script measures calibration for three scores on the same nine fold-units:

    age        logistic regression on age alone, fitted on the training split
    image      the fused image model, p = alpha / (alpha + beta)
    image+age  z + [logit pi(a) - logit ybar_train]  (the auc_table sum, not retrained)

Per unit and score: AUC, ECE (10 equal bins, same definition as eval.metrics),
NLL, Brier, the calibration intercept (logit offset with slope fixed at 1, i.e.
how far the score would have to move to match the target labels) and the
calibration slope (logistic recalibration a + s * logit p). The image scores are
averaged over the three seeds; the age model has no seed, so its spread is a
bootstrap over test subjects. No target label is used to fit any score.

The run of record is notebooks/08_age_baseline_calibration.ipynb, which imports
this module and writes the json / md / figure. The CLI prints the same tables:

    OPENIDH_PATHS=configs/paths.hpc.yaml uv run --frozen papermill \
        notebooks/08_age_baseline_calibration.ipynb notebooks/08_age_baseline_calibration.ipynb
    uv run --frozen python scripts/age_baseline_calibration.py --paths-file configs/paths.hpc.yaml

CPU only, under a minute. Runs without runs/*/predictions.csv too (age rows only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from age_prior_eval import _fit_age_model, _lo, subject_table  # noqa: E402
from openidh_model.eval.metrics import _ece  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402

# same units, files and run tags as age_prior_eval.auc_table
UNITS = [("LOSO-A", "splits_loso_foldA.csv", None, "foldA", "OOD"),
         ("LOSO-B", "splits_loso_foldB.csv", None, "foldB", "OOD"),
         ("vendor Philips", "splits_vendor_philips.csv", None, "vendor", "OOD"),
         ("field 3T", "splits_field.csv", None, "field", "OOD")]
UNITS += [(f"random f{i}", "splits_random_5fold.csv", i, f"rand{i}", "in-dist")
          for i in range(5)]
SCORES = ("age", "image", "image+age")
_EPS = 1e-7


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _nll(p, y):
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_intercept(p, y) -> float:
    """Logit offset b (slope fixed at 1) that maximises the target likelihood.

    Positive b = the score under-predicts the target and would have to move up.
    """
    z = _logit(p)
    f = lambda b: _nll(1 / (1 + np.exp(-(z + b))), y)  # noqa: E731
    return float(minimize_scalar(f, bounds=(-6, 6), method="bounded").x)


def calibration_slope(p, y) -> tuple[float, float]:
    """(a, s) of the logistic recalibration a + s * logit p. Perfect = (0, 1)."""
    z = _logit(p)
    f = lambda w: _nll(1 / (1 + np.exp(-(w[0] + w[1] * z))), y)  # noqa: E731
    w = minimize(f, x0=np.array([0.0, 1.0]), method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000}).x
    return float(w[0]), float(w[1])


def score_metrics(p, y) -> dict:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    a, s = calibration_slope(p, y)
    return {"auc": float(roc_auc_score(y, p)), "ece": _ece(p, y), "nll": _nll(p, y),
            "brier": float(np.mean((p - y) ** 2)), "mean_p": float(p.mean()),
            "prevalence": float(y.mean()), "cal_intercept": calibration_intercept(p, y),
            "recal_a": a, "cal_slope": s}


def reliability(p, y, n_bins: int = 10) -> list[dict]:
    """Same bins as _ece: (lo, hi], empty bins dropped."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return out


def bootstrap_ece(p, y, reps: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    v = [_ece(p[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(reps))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def evaluate(paths, runs_dir: Path, reps: int) -> list[dict]:
    subj = subject_table(paths).dropna(subset=["age"])
    out = []
    for label, f, fid, tag, kind in UNITS:
        sp = pd.read_csv(paths.splits_dir / f)
        if fid is not None:
            sp = sp[sp["fold_id"].astype(str) == str(fid)]
        sp = sp.merge(subj[["subject_id", "age"]], on="subject_id", how="left").dropna(subset=["age"])
        sp["mut"] = (sp["idh"].astype(str).str.strip() == "Mut").astype(int)
        tr, te = sp[sp.split_role == "train"], sp[sp.split_role == "test"]
        pi_of = _fit_age_model(tr["age"].to_numpy(float), tr["mut"].to_numpy(int), False)
        lo_train = _lo(float(tr["mut"].mean()))

        y_age = te["mut"].to_numpy(float)
        p_age = pi_of(te["age"].to_numpy(float))
        row = {"label": label, "kind": kind, "n_test": len(te),
               "pi_train": float(tr["mut"].mean()), "pi_test": float(y_age.mean()),
               "age": score_metrics(p_age, y_age),
               "age_ece_ci95": bootstrap_ece(p_age, y_age, reps),
               "reliability": {"age": reliability(p_age, y_age)}, "seeds": []}

        rel_img, rel_comb = [], []
        for seed in range(3):
            pf = runs_dir / f"{tag}_s{seed}" / "predictions.csv"
            if not pf.is_file():
                continue
            t = pd.read_csv(pf)
            t = t[t.role == "test"].merge(subj[["subject_id", "age"]], on="subject_id", how="left")
            y = t["y_true"].to_numpy(float)
            z = np.log(t["alpha"] / t["beta"]).to_numpy()
            za = _logit(pi_of(t["age"].to_numpy(float))) - lo_train
            p_img = 1 / (1 + np.exp(-z))                       # = alpha / (alpha + beta)
            p_comb = 1 / (1 + np.exp(-(z + za)))
            row["seeds"].append({"seed": seed, "image": score_metrics(p_img, y),
                                 "image+age": score_metrics(p_comb, y)})
            rel_img.append(reliability(p_img, y))
            rel_comb.append(reliability(p_comb, y))
        if row["seeds"]:
            for k in ("image", "image+age"):
                keys = row["seeds"][0][k].keys()
                row[k] = {m: float(np.mean([s[k][m] for s in row["seeds"]])) for m in keys}
                row[k + "_sd"] = {m: float(np.std([s[k][m] for s in row["seeds"]])) for m in keys}
            row["reliability"]["image"] = rel_img        # one list per seed
            row["reliability"]["image+age"] = rel_comb
        out.append(row)
    return out


def summary(rows: list[dict]) -> dict:
    """Unit-mean per condition, as in the manuscript (OOD = 4 units, in-dist = 5)."""
    out = {}
    for kind in ("OOD", "in-dist"):
        rs = [r for r in rows if r["kind"] == kind]
        out[kind] = {sc: {m: float(np.mean([r[sc][m] for r in rs]))
                          for m in ("auc", "ece", "nll", "brier", "cal_intercept", "cal_slope")}
                     for sc in SCORES if all(sc in r for r in rs)}
    return out


def tables_md(rows: list[dict], summ: dict) -> str:
    L = []
    present = [sc for sc in SCORES if all(sc in r for r in rows)]
    L.append("## 較正：年齢のみ・画像モデル・画像 + 年齢（test。画像は 3 seed 平均）\n")
    L.append("| ユニット | 条件 | n | 有病率 train→test | スコア | AUC | ECE | NLL | Brier "
             "| 平均予測 | 較正切片 | 較正スロープ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for sc in present:
            m = r[sc]
            ece = f"{m['ece']:.3f}"
            if sc == "age":
                lo, hi = r["age_ece_ci95"]
                ece += f" [{lo:.3f}–{hi:.3f}]"
            else:
                ece += f" ±{r[sc + '_sd']['ece']:.3f}"
            L.append(f"| {r['label']} | {r['kind']} | {r['n_test']} | "
                     f"{r['pi_train']*100:.1f}%→{r['pi_test']*100:.1f}% | {sc} | {m['auc']:.3f} | "
                     f"{ece} | {m['nll']:.3f} | {m['brier']:.3f} | {m['mean_p']*100:.1f}% | "
                     f"{m['cal_intercept']:+.2f} | {m['cal_slope']:.2f} |")
    L.append("\n## 条件ごとの平均（ユニット平均）\n")
    L.append("| 条件 | スコア | AUC | ECE | NLL | Brier | 較正切片 | 較正スロープ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for kind, d in summ.items():
        for sc, m in d.items():
            L.append(f"| {kind} | {sc} | {m['auc']:.3f} | {m['ece']:.3f} | {m['nll']:.3f} | "
                     f"{m['brier']:.3f} | {m['cal_intercept']:+.2f} | {m['cal_slope']:.2f} |")
    L.append("\n較正切片：傾き 1 のままロジットに足すべき量（正 = 過小評価）。"
             "較正スロープ：1 より小さい = 過信、大きい = 控えめすぎ。"
             "年齢の ECE の [ ] は test 症例のブートストラップ 95% 区間、画像の ± は seed 間 sd。")
    return "\n".join(L)


def auc_crosscheck(rows: list[dict], prev: Path = _ROOT / "results" / "age_prior.json"):
    """Max |AUC difference| against the committed auc_table (same split, fit and predictions)."""
    if not prev.is_file():
        return None
    old = {x["label"]: x for x in json.loads(prev.read_text())["auc_table"]}
    d = [abs(r["age"]["auc"] - old[r["label"]]["auc_age"]) for r in rows if r["label"] in old]
    d += [abs(r["image"]["auc"] - old[r["label"]]["auc_image"])
          for r in rows if r["label"] in old and "image" in r]
    return float(max(d))


def plot(rows: list[dict], path: Path | None = None):
    import matplotlib.pyplot as plt

    col = {"age": "#eb6834", "image": "#2a78d6", "image+age": "#1baf7a"}   # categorical slots 2, 1, 3
    name = {"age": "age only", "image": "image model", "image+age": "image + age"}
    panels = [r for r in rows if r["kind"] == "OOD"] + [None]               # last = in-dist pooled
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.4), sharey=True)
    for ax, r in zip(axes, panels):
        group = [r] if r else [x for x in rows if x["kind"] == "in-dist"]
        title = r["label"] if r else "in-dist (5 folds)"
        ax.plot([0, 1], [0, 1], color="#8a8a86", lw=1, ls=(0, (2, 2)))
        for sc in SCORES:
            bins = []
            for x in group:
                rel = x["reliability"].get(sc)
                if rel is None:
                    continue
                bins += rel if sc == "age" else [b for seed in rel for b in seed]
            if not bins:
                continue
            # pool bins by edge, weighting by count
            agg = {}
            for b in bins:
                a = agg.setdefault(b["lo"], [0, 0.0, 0.0])
                a[0] += b["n"]; a[1] += b["mean_p"] * b["n"]; a[2] += b["frac_pos"] * b["n"]
            xs = sorted(agg)
            ax.plot([agg[k][1] / agg[k][0] for k in xs], [agg[k][2] / agg[k][0] for k in xs],
                    color=col[sc], lw=2, marker="o", ms=4, label=name[sc])
        ece = " / ".join(f"{np.mean([x[sc]['ece'] for x in group]):.3f}"
                         for sc in SCORES if all(sc in x for x in group))
        ax.set_title(f"{title}\nECE {ece}", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.grid(color="#e6e5e0", lw=.6)
        ax.set_xlabel("mean predicted probability", fontsize=8)
    axes[0].set_ylabel("observed IDH-mutant fraction", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Reliability: age only vs image model vs image + age "
                 "(ECE order: age / image / image+age)", fontsize=10)
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"wrote {path}")
    return fig


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_ROOT / "runs"))
    ap.add_argument("--paths-file", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--fig", default=None)
    ap.add_argument("--reps", type=int, default=1000, help="bootstrap reps for the age ECE")
    a = ap.parse_args()
    paths = load_paths(a.paths_file or "configs/paths.local.yaml")

    rows = evaluate(paths, Path(a.runs_dir), a.reps)
    missing = [r["label"] for r in rows if not r["seeds"]]
    if missing:
        print(f"[warn] predictions.csv が無いユニット（年齢のみ出力）: {missing}\n")
    summ = summary(rows)
    print(tables_md(rows, summ))

    d = auc_crosscheck(rows)
    if d is not None:
        print(f"\n検算: age_prior.json の AUC との最大差 {d:.2e}（同じ分割・同じ予測なので 0 のはず）")

    if a.json:
        Path(a.json).write_text(json.dumps({"units": rows, "summary": summ}, indent=2,
                                           default=float, ensure_ascii=False))
        print(f"wrote {a.json}")
    if a.fig:
        plot(rows, Path(a.fig))


if __name__ == "__main__":
    main()
