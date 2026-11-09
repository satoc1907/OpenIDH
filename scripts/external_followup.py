"""External cohorts (EGD, Japan), after eval_external.py — no torch, no GPU.

Two questions, both answered from the external predictions.csv alone:

1. Output matching on real external sites. The 06/07 machinery (output_matching,
   gated_correction) is applied unchanged: each LOSO run keeps the covariate prior
   pi(age, suspicion) fitted on its own training sites and its own inner-val
   anchor, and the external cohort plays the role of the new site.
     evaluation 1  whole cohort: raw / ratio / output matching / oracle / all-label MLE
     evaluation 2  the first N cases (N = 20, 50, 100, 200; 50 draws), ungated and
                   with the significance gate (k = 2.5), scored on the remainder
   Suspicion = high grade (4) vs low (2/3), from the cohort's WHO grade column
   (`who_grade`, WHO 2016 for both cohorts by default; Japan also has
   `who_grade_2021`, but the 2021 grade is partly decided by IDH-wt molecular
   markers, so it leaks the label into pi — use it only as a sensitivity check).
   Cases without age cannot enter pi(age, ...) and are left out of part 1
   (EGD: 42 of 467); the table reports the n actually used.

2. Is the uncertainty still informative off-site? For a Beta(alpha, beta) output,
   S = alpha + beta is the evidence: small S = wide distribution (heavy tails).
   If the design works, wrong predictions should have smaller S than right ones.
   Per run x cohort (and the run's own LOSO test, for reference): median S of
   right vs wrong (p > 0.5, raw and after output matching), the AUROC of S for
   telling right from wrong, Spearman(S, |y - p|), and AUC within S tertiles.

    uv run python scripts/external_followup.py            # OPENIDH_PATHS=configs/paths.hpc.yaml
    uv run python scripts/external_followup.py --grade-col who_grade_2021 --cohorts Japan

Writes <external_eval>/followup{_<grade-col>}.{json,md} — aggregates only.
notebooks/09_external_validation.ipynb runs the same analyse() and adds the figures
(plot_reliability, plot_n_curve, plot_uncertainty) and results/external_validation.*.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openidh_model.data.metadata import EXTERNAL_COHORTS  # noqa: E402
from openidh_model.eval import gated_correction as gc  # noqa: E402
from openidh_model.eval import output_matching as om  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402

LOSO_RUNS = [f"{u}_s{s}" for u in ("foldA", "foldB") for s in range(3)]
NS = (20, 50, 100, 200)
K = 2.5                # the gate under review (07: k = 2.5, N_min = 20 = the smallest N here)
_EPS = 1e-7


def cohort_covariates(meta_dir: Path, cohort: str, grade_col: str) -> pd.DataFrame:
    t = pd.read_csv(meta_dir / EXTERNAL_COHORTS[cohort][1], dtype={"subject_id": str},
                    keep_default_na=False)
    if grade_col not in t:
        sys.exit(f"{cohort}: no column {grade_col!r}")
    grade = pd.to_numeric(t[grade_col], errors="coerce")
    grade = grade.where(grade.isin([2, 3, 4]))
    return pd.DataFrame({"subject_id": t["subject_id"].str.strip(),
                         "age": pd.to_numeric(t["age"], errors="coerce"),
                         "susp": om.suspicion_from_grade(grade)})


def external_runs(src: list, pred: pd.DataFrame, covs: dict) -> tuple[list, dict]:
    """RunData with the external cohort as `test`. Prior and val stay the run's own."""
    out, dropped = [], {}
    for rd in src:
        for cohort, cov in covs.items():
            t = pred[(pred.run == rd.run) & (pred.role == cohort)].merge(cov, on="subject_id", how="left")
            if t.empty:
                continue
            dropped[cohort] = int(t["age"].isna().sum())
            t = t[t["age"].notna()].reset_index(drop=True)
            t = t.assign(z=np.log(np.clip(t["alpha"], _EPS, None)) - np.log(np.clip(t["beta"], _EPS, None)),
                         y=t["y_true"].astype(float))
            t["pi_cov"] = rd.prior.predict(t["age"], t["susp"])
            t["pi_age"] = rd.prior.predict(t["age"])
            out.append(om.RunData(rd.run, f"{rd.unit} → {cohort}", rd.seed, rd.prior, rd.val,
                                  t[["subject_id", "z", "y", "age", "susp", "pi_cov", "pi_age"]]))
    return out, dropped


# ------------------------------------------------------------- part 2: uncertainty

def uncertainty(p: np.ndarray, S: np.ndarray, y: np.ndarray, b: float = 0.0) -> dict:
    """Right vs wrong at 0.5 after adding intercept b to the logit (S unchanged)."""
    if b:
        p = expit(np.log(np.clip(p, _EPS, 1 - _EPS) / np.clip(1 - p, _EPS, 1)) + b)
    right = (p > 0.5).astype(int) == y.astype(int)
    out = {"n": int(len(y)), "acc": float(right.mean()),
           "S_median": float(np.median(S)),
           "S_right": float(np.median(S[right])) if right.any() else np.nan,
           "S_wrong": float(np.median(S[~right])) if (~right).any() else np.nan,
           "n_wrong": int((~right).sum())}
    # AUROC of S for "right": 0.5 = S says nothing, > 0.5 = smaller S when wrong
    out["auroc_S_right"] = (float(roc_auc_score(right, S))
                            if 0 < right.sum() < len(right) else np.nan)
    out["spearman_S_err"] = float(spearmanr(S, np.abs(y - p)).statistic)
    # discrimination within S tertiles: low S should be the hard cases
    cuts = np.quantile(S, [1 / 3, 2 / 3])
    terc = np.digitize(S, cuts)
    for i, name in enumerate(("low", "mid", "high")):
        m = terc == i
        out[f"auc_S_{name}"] = (float(roc_auc_score(y[m], p[m]))
                                if m.sum() and 0 < y[m].sum() < m.sum() else np.nan)
    return out


def part2(pred: pd.DataFrame, runs_dir: Path, runs: list, offsets: dict) -> pd.DataFrame:
    rows = []
    for run in runs:
        own = pd.read_csv(runs_dir / run / "predictions.csv", dtype={"subject_id": str})
        own = own[own.role == "test"].assign(role="own test")
        for cohort, g in pd.concat([own, pred[pred.run == run]]).groupby("role", sort=False):
            p, S, y = g["p_hat"].to_numpy(), g["S"].to_numpy(), g["y_true"].to_numpy()
            rows.append({"run": run, "cohort": cohort, "after": "raw", **uncertainty(p, S, y)})
            b = offsets.get((run, cohort))
            if b is not None:
                rows.append({"run": run, "cohort": cohort, "after": "om", **uncertainty(p, S, y, b)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ report

def _ms(s: pd.Series, f="{:.3f}") -> str:
    s = s.dropna()
    if s.empty:
        return "—"
    if len(s) == 1:
        return f.format(s.mean())
    return f.format(s.mean()) + " ± " + f.replace("+", "").format(s.std(ddof=1))


def df_to_md(df: pd.DataFrame) -> str:
    """Pipe table without tabulate (not in the lock)."""
    d = df.reset_index()
    rows = [list(map(str, d.columns)), ["---"] * len(d.columns)] + d.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def report(full: pd.DataFrame, gsum: pd.DataFrame, unc: pd.DataFrame, dropped: dict,
           grade_col: str, reps: int) -> str:
    L = ["# External cohorts — output matching and uncertainty", "",
         f"Suspicion from `{grade_col}` (grade 4 = high). Mean ± SD over the 3 seeds of each "
         "LOSO unit. Cases without age are left out of part 1: "
         + ", ".join(f"{c} {n}" for c, n in dropped.items()) + ".", "",
         "## 1. Output matching, whole cohort", "",
         "| unit → cohort | n | true rate | π̂_cov | method | b | ECE | NLL | Brier |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (unit, method), g in full.groupby(["unit", "method"], sort=False):
        L.append(f"| {unit} | {g['n'].iloc[0]} | {g['pi_true'].mean():.3f} | {g['pi_cov_test'].mean():.3f} "
                 f"| {om.METHODS_FULL[method]} | {_ms(g['b'], '{:+.2f}')} | {_ms(g['ece'])} "
                 f"| {_ms(g['nll'])} | {_ms(g['brier'])} |")
    clipped = full[full.status != "ok"]
    if not clipped.empty:
        L += ["", f"Clipped solves: {len(clipped)} ({', '.join(clipped.run + '/' + clipped.method)})"]

    L += ["", f"## 2. The first N cases — output matching, ungated and gated (k = {K})", "",
          f"ECE on the remaining cases, mean over 3 seeds × {reps} draws. P(harm) = P(ECE worse than "
          "uncorrected by > 0.01). `label` uses the N cases' labels (the labelled baseline).", "",
          "| unit → cohort | N | uncorrected | om | om gated | fire rate | P(harm) gated | label |",
          "|---|---|---|---|---|---|---|---|"]
    s = gsum
    for (unit, n), g in s.groupby(["unit", "n"], sort=False):
        o = g[g.method == "om"].set_index("gate")
        lab = g[(g.method == "label") & (g.gate == "none")]
        L.append(f"| {unit} | {n} | {o.loc['none', 'ece_raw']:.3f} | {o.loc['none', 'ece_mean']:.3f} "
                 f"| {o.loc[str(K), 'ece_mean']:.3f} | {o.loc[str(K), 'fire_rate']:.0%} "
                 f"| {o.loc[str(K), 'p_harm_tol']:.0%} | {lab['ece_mean'].iloc[0]:.3f} |")

    L += ["", "## 3. Uncertainty: is S smaller when the prediction is wrong?", "",
          "S = α + β (evidence). Small S = wide Beta = heavy tails. Right/wrong at p̂ > 0.5, "
          "`raw` = as predicted, `om` = after the whole-cohort output-matching intercept. "
          "AUROC(S) > 0.5 means wrong cases have smaller S. Spearman(S, |y − p̂|) < 0 means "
          "the larger the error, the smaller S.", "",
          "| unit | cohort | after | n | accuracy | wrong | S median | S right | S wrong "
          "| AUROC(S) | Spearman | AUC low S | AUC mid S | AUC high S |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    unc = unc.assign(unit=unc.run.str.slice(0, 5).map({"foldA": "LOSO-A", "foldB": "LOSO-B"}))
    for (unit, cohort, after), g in unc.groupby(["unit", "cohort", "after"], sort=False):
        L.append(f"| {unit} | {cohort} | {after} | {g['n'].iloc[0]} | {_ms(g['acc'])} "
                 f"| {g['n_wrong'].mean():.0f} | {_ms(g['S_median'], '{:.1f}')} "
                 f"| {_ms(g['S_right'], '{:.1f}')} | {_ms(g['S_wrong'], '{:.1f}')} "
                 f"| {_ms(g['auroc_S_right'])} | {_ms(g['spearman_S_err'], '{:+.3f}')} "
                 f"| {_ms(g['auc_S_low'])} | {_ms(g['auc_S_mid'])} | {_ms(g['auc_S_high'])} |")
    return "\n".join(L) + "\n"


def analyse(paths, runs=LOSO_RUNS, cohorts=None, grade_col="who_grade", reps=50, seed=0,
            eval_dir=None, runs_dir=None) -> dict:
    """Everything parts 1 and 2 need, in one call (the CLI and notebook 09 share it)."""
    cohorts = list(cohorts or EXTERNAL_COHORTS)
    if paths.external_metadata_dir is None:
        raise ValueError("external_metadata_dir is not set in the paths YAML")
    eval_dir = Path(eval_dir or paths.external_metadata_dir.parent / "external_eval")
    runs_dir = Path(runs_dir or paths.output_dir)

    pred = pd.read_csv(eval_dir / "predictions.csv", dtype={"subject_id": str, "active_mask": str})
    pred = pred[pred.run.isin(runs) & pred.role.isin(cohorts)]
    covs = {c: cohort_covariates(paths.external_metadata_dir, c, grade_col) for c in cohorts}

    src = [rd for rd in om.load_runs(paths, runs_dir) if rd.run in runs]
    missing = sorted(set(runs) - {rd.run for rd in src})
    if missing:
        raise FileNotFoundError(f"no runs/<name>/predictions.csv (val + test) for: {missing}")
    ext, dropped = external_runs(src, pred, covs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # clipped solves are counted in the table instead
        full = om.evaluate_full(ext)
        n_of = {(rd.run, rd.unit): len(rd.test) for rd in ext}
        full["n"] = [n_of[(r, u)] for r, u in zip(full.run, full.unit)]
        gcurve = gc.gated_curve(ext, ns=NS, reps=reps, seed=seed, ks=(K,))
    gsum = gc.summarise_gated(gcurve, ks=(K,))

    # whole-cohort om intercept per (run, cohort), for the "after om" views
    om_b = {(r.run, r.unit.split(" → ")[1]): float(b) for r in ext
            for b in full[(full.run == r.run) & (full.unit == r.unit) & (full.method == "c_om_cov")]["b"]}
    unc = part2(pred, runs_dir, runs, om_b)
    return {"pred": pred, "ext": ext, "dropped": dropped, "full": full, "gsum": gsum,
            "om_b": om_b, "unc": unc, "eval_dir": eval_dir, "runs_dir": runs_dir,
            "params": {"runs": list(runs), "cohorts": cohorts, "grade_col": grade_col,
                       "reps": reps, "seed": seed, "k": K, "ns": list(NS)}}


def to_json(a: dict) -> dict:
    return {"params": a["params"], "dropped_no_age": a["dropped"],
            "full": a["full"].to_dict(orient="records"),
            "gated": a["gsum"].to_dict(orient="records"),
            "uncertainty": a["unc"].to_dict(orient="records")}


# ------------------------------------------------------------------------ figures
# English labels: the HPC image has no CJK font.

UNITS = ("LOSO-A", "LOSO-B")


def _reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return np.array(out)


def plot_reliability(a: dict, path):
    """Unit x cohort panels; raw vs whole-cohort output matching, 3 seeds pooled."""
    import matplotlib.pyplot as plt
    cohorts = a["params"]["cohorts"]
    fig, axes = plt.subplots(len(UNITS), len(cohorts), figsize=(4.2 * len(cohorts), 4.2 * len(UNITS)),
                             squeeze=False)
    for i, unit in enumerate(UNITS):
        for j, c in enumerate(cohorts):
            ax = axes[i, j]
            rds = [r for r in a["ext"] if r.unit == f"{unit} → {c}"]
            if not rds:
                ax.set_visible(False)
                continue
            z = np.concatenate([r.test["z"] for r in rds])
            y = np.concatenate([r.test["y"] for r in rds])
            b = np.concatenate([np.full(len(r.test), a["om_b"][(r.run, c)]) for r in rds])
            ax.plot([0, 1], [0, 1], color="0.7", lw=1)
            for lab, p, col in (("uncorrected", expit(z), "#c0392b"),
                                ("output matching", expit(z + b), "#2471a3")):
                r = _reliability(p, y)
                ax.plot(r[:, 0], r[:, 1], "o-", color=col, label=lab, ms=4)
            ax.axhline(y.mean(), color="0.5", ls=":", lw=1)
            ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{unit} → {c} (n={len(rds[0].test)})",
                   xlabel="predicted P(IDH-mut)", ylabel="observed rate")
            ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def plot_n_curve(a: dict, path):
    """ECE on the remaining cases vs N: uncorrected, om, om gated, labelled."""
    import matplotlib.pyplot as plt
    s = a["gsum"]
    units = list(dict.fromkeys(s["unit"]))
    fig, axes = plt.subplots(1, len(units), figsize=(4 * len(units), 3.6), squeeze=False, sharey=True)
    for ax, unit in zip(axes[0], units):
        g = s[s.unit == unit]
        om_none = g[(g.method == "om") & (g.gate == "none")].sort_values("n")
        om_k = g[(g.method == "om") & (g.gate == str(K))].sort_values("n")
        lab = g[(g.method == "label") & (g.gate == "none")].sort_values("n")
        ax.plot(om_none.n, om_none.ece_raw, "s--", color="#c0392b", label="uncorrected")
        ax.plot(om_none.n, om_none.ece_mean, "o-", color="#2471a3", label="output matching")
        ax.plot(om_k.n, om_k.ece_mean, "o:", color="#2471a3", mfc="white", label=f"om gated (k={K})")
        ax.plot(lab.n, lab.ece_mean, "^-", color="#7d8c3a", label="labelled N")
        ax.set(xscale="log", title=unit, xlabel="N cases used")
        ax.set_xticks(om_none.n, [str(n) for n in om_none.n])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    axes[0, 0].set_ylabel("ECE on the remaining cases")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def plot_uncertainty(a: dict, path):
    """S of right vs wrong predictions (raw p > 0.5), 3 seeds pooled, beside the own LOSO test."""
    import matplotlib.pyplot as plt
    pred, runs_dir = a["pred"], a["runs_dir"]
    cols = ["own test"] + a["params"]["cohorts"]
    fig, axes = plt.subplots(1, len(UNITS), figsize=(5.5 * len(UNITS), 3.8), squeeze=False, sharey=True)
    for ax, (prefix, unit) in zip(axes[0], (("foldA", "LOSO-A"), ("foldB", "LOSO-B"))):
        runs = [r for r in a["params"]["runs"] if r.startswith(prefix)]
        own = pd.concat([pd.read_csv(runs_dir / r / "predictions.csv").query("role == 'test'")
                         .assign(role="own test") for r in runs])
        d = pd.concat([own, pred[pred.run.isin(runs)]])
        data, pos, colors = [], [], []
        for i, c in enumerate(cols):
            g = d[d.role == c]
            right = (g.p_hat > 0.5).astype(int) == g.y_true
            for k, (m, col) in enumerate(((right, "#2471a3"), (~right, "#c0392b"))):
                data.append(g.S[m].to_numpy())
                pos.append(i * 3 + k)
                colors.append(col)
        bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], colors):
            patch.set(facecolor=col, alpha=0.5)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(cols))], cols)
        ax.set_title(f"{unit}: evidence S = α + β (blue right, red wrong)")
    axes[0, 0].set_ylabel("S (smaller = wider Beta)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=LOSO_RUNS)
    ap.add_argument("--cohorts", nargs="*", default=list(EXTERNAL_COHORTS))
    ap.add_argument("--grade-col", default="who_grade")
    ap.add_argument("--eval-dir", default=None, help="default: <external_metadata_dir>/../external_eval")
    ap.add_argument("--runs-dir", default=None, help="default: paths output_dir (runs/)")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paths-file", default=None, help="defaults to $OPENIDH_PATHS")
    args = ap.parse_args()

    a = analyse(load_paths(args.paths_file), args.runs, args.cohorts, args.grade_col,
                args.reps, args.seed, args.eval_dir, args.runs_dir)
    tag = "" if args.grade_col == "who_grade" else f"_{args.grade_col}"
    md = report(a["full"], a["gsum"], a["unc"], a["dropped"], args.grade_col, args.reps)
    (a["eval_dir"] / f"followup{tag}.md").write_text(md)
    (a["eval_dir"] / f"followup{tag}.json").write_text(json.dumps(to_json(a), indent=1, default=float))
    print(md)
    print(f"-> {a['eval_dir']}/followup{tag}.{{md,json}}")


if __name__ == "__main__":
    main()
