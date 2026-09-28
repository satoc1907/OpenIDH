"""Output matching: a label-free intercept that absorbs every source of shift.

Experiment A showed the LOSO-B offset is a tug of war — case mix pushes the
criterion up, the scanner pushes it down — and that on field 3T the scanner alone
moves the fused logit by -0.33 with prevalence unchanged. The ratio method
(indirect standardisation: look both populations up in the same covariate model)
only sees the case-mix half.

Output matching takes the prevalence from covariates and lets the model's own
outputs measure how far off it currently is:

    find b  such that  (1/N) sum_i sigmoid(z_i + b) = pi_cov

z_i = log(alpha_i / beta_i) is the uncorrected fused logit; pi_cov is the target
prevalence predicted by pi(age, suspicion) fitted on the training sites. The left
side is strictly increasing in b, so brentq finds the unique root. Unlike EM the
prevalence never passes through the model, so there is no loop.

Consistency check built into the method: anchored at the TRUE prevalence, the
equation is exactly the score equation of the intercept-only MLE on all labels
(sum sigmoid(z_i + b) = sum y_i), so the two must agree numerically.

No torch: this runs on the analysis machine from predictions.csv alone.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from ..data.metadata import lookup_age_sex, lookup_grade
from .metrics import compute_metrics

BRACKET = (-10.0, 10.0)
NS = (5, 10, 20, 30, 50, 100, 200)
_EPS = 1e-7

# split_file -> unit label (same labels as scripts/aggregate_results.py)
UNIT_OF_SPLIT = {
    "splits_loso_foldA.csv": "LOSO-A",
    "splits_loso_foldB.csv": "LOSO-B",
    "splits_field.csv": "field 3T",
    "splits_vendor_philips.csv": "vendor Philips",
    "splits_random_5fold.csv": "in-dist",
}
OOD_UNITS = ("LOSO-A", "LOSO-B", "field 3T", "vendor Philips")


def logit(p) -> float:
    p = float(np.clip(p, _EPS, 1 - _EPS))
    return float(np.log(p / (1 - p)))


# ---------------------------------------------------------------- the estimator

def solve_offset(z, target: float, bracket=BRACKET) -> tuple[float, str]:
    """b with mean(sigmoid(z + b)) == target. Returns (b, status).

    status is "ok", or "clip_lo" / "clip_hi" when the target lies outside what the
    bracket can reach — b is then pinned to that end and a warning is raised. A
    pinned run is kept and reported, never dropped.
    """
    z = np.asarray(z, dtype=np.float64)
    lo, hi = bracket
    f = lambda b: float(expit(z + b).mean()) - target
    flo, fhi = f(lo), f(hi)
    if flo >= 0:
        warnings.warn(f"output matching: target {target:.4f} <= reachable min "
                      f"{target + flo:.4f}; b clipped to {lo}")
        return lo, "clip_lo"
    if fhi <= 0:
        warnings.warn(f"output matching: target {target:.4f} >= reachable max "
                      f"{target + fhi:.4f}; b clipped to {hi}")
        return hi, "clip_hi"
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)), "ok"


def label_intercept_mle(z, y, bracket=BRACKET) -> float:
    """Intercept-only MLE on labelled cases, by direct NLL minimisation.

    Deliberately NOT written via solve_offset: it is the independent reference
    that output matching anchored at the true prevalence has to reproduce.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def nll(b):
        p = np.clip(expit(z + b), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    r = minimize_scalar(nll, bounds=bracket, method="bounded",
                        options={"xatol": 1e-10})
    return float(r.x)


def ratio_offset(pi_target: float, pi_source: float) -> float:
    """Indirect standardisation: both populations through the same pi model."""
    return logit(pi_target) - logit(pi_source)


def calib_metrics(z, y, b: float = 0.0) -> dict:
    """ECE / Brier / NLL / AUC and sens/spec at 0.5 after adding intercept b.

    Goes through compute_metrics so ECE is the same 10-bin estimator used in every
    earlier report. Only p = sigmoid(z + b) matters to these metrics, so the Beta
    is passed with S = 1.
    """
    p = expit(np.asarray(z, dtype=np.float64) + b)
    y = np.asarray(y, dtype=np.float64)
    m = compute_metrics(p, 1.0 - p, y)
    pos, neg = y == 1, y == 0
    m["sens"] = float(((p > .5) & pos).sum() / max(pos.sum(), 1))
    m["spec"] = float(((p <= .5) & neg).sum() / max(neg.sum(), 1))
    return {k: m[k] for k in ("ece", "brier", "nll", "auc", "sens", "spec")}


# ------------------------------------------------------ covariate prevalence pi

def suspicion_from_grade(grade) -> np.ndarray:
    """Binary pre-operative read: 1 = high grade (4), 0 = low (2/3), nan = unknown.
    The same encoding as level B in scripts/grade_prior_eval.py."""
    g = np.asarray(grade, dtype=np.float64)
    return np.where(np.isnan(g), np.nan, (g == 4).astype(float))


def flip_balanced(susp, rate: float, rng) -> np.ndarray:
    """Misread `rate` of the known answers, preserving the high/low marginal
    (per-class rates with p*a == (1-p)*b), as level Cb of grade_prior_eval."""
    s = np.asarray(susp, dtype=np.float64).copy()
    obs = ~np.isnan(s)
    if not obs.any():
        return s
    high = obs & (s == 1)
    p = high[obs].mean()
    low_rate = min(1.0, rate * p / max(1 - p, 1e-9))
    r = np.where(high, rate, low_rate)
    roll = obs & (rng.random(len(s)) < r)
    s[roll] = 1.0 - s[roll]
    return s


def mask_unknown(susp, frac: float, rng) -> np.ndarray:
    """Set `frac` of the known answers to unknown (nan)."""
    s = np.asarray(susp, dtype=np.float64).copy()
    obs = np.flatnonzero(~np.isnan(s))
    k = int(round(frac * len(obs)))
    if k:
        s[rng.choice(obs, size=k, replace=False)] = np.nan
    return s


@dataclass
class CovariatePrior:
    """pi(age, suspicion) and the age-only fallback, both fitted on training sites.

    Essentially unpenalised logistic regression (C=1e6), as in age_prior_eval /
    grade_prior_eval. A case whose suspicion is unknown gets the age-only pi.
    """
    full: LogisticRegression
    age_only: LogisticRegression

    @classmethod
    def fit(cls, age, susp, y) -> "CovariatePrior":
        age = np.asarray(age, dtype=np.float64)
        susp = np.asarray(susp, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        k = ~np.isnan(susp)
        full = LogisticRegression(max_iter=5000, C=1e6).fit(
            np.column_stack([age[k], susp[k]]), y[k])
        age_only = LogisticRegression(max_iter=5000, C=1e6).fit(age.reshape(-1, 1), y)
        return cls(full, age_only)

    def predict(self, age, susp=None) -> np.ndarray:
        """Per-case pi_i. susp=None means age only for everyone."""
        age = np.asarray(age, dtype=np.float64)
        out = self.age_only.predict_proba(age.reshape(-1, 1))[:, 1]
        if susp is None:
            return out
        susp = np.asarray(susp, dtype=np.float64)
        k = ~np.isnan(susp)
        if k.any():
            out = out.copy()
            out[k] = self.full.predict_proba(np.column_stack([age[k], susp[k]]))[:, 1]
        return out


# ----------------------------------------------------------------- data loading

def subject_covariates(paths, split_file: str = "splits_loso_foldA.csv") -> pd.DataFrame:
    """subject_id -> site, age, grade, suspicion, mut (one row per subject)."""
    sp = pd.read_csv(Path(paths.splits_dir) / split_file).drop_duplicates("subject_id")
    rows = []
    for _, r in sp.iterrows():
        age, _sex = lookup_age_sex(paths.metadata_dir, r["site"], r["subject_id"])
        rows.append({"subject_id": r["subject_id"], "site": r["site"], "age": age,
                     "grade": lookup_grade(paths.metadata_dir, r["site"], r["subject_id"]),
                     "mut": int(str(r["idh"]).strip() == "Mut")})
    d = pd.DataFrame(rows)
    d["susp"] = suspicion_from_grade(d["grade"])
    return d


@dataclass
class RunData:
    run: str
    unit: str
    seed: int
    prior: CovariatePrior
    val: pd.DataFrame     # z, y, age, susp, pi_cov, pi_age
    test: pd.DataFrame


def load_runs(paths, runs_dir) -> list[RunData]:
    """Every production run (train.yaml) with test/val predictions joined to
    covariates, and the covariate prior fitted on that split's train role."""
    runs_dir = Path(runs_dir)
    cov = subject_covariates(paths)
    priors: dict = {}
    out = []
    for rj in sorted(runs_dir.glob("*/result.json")):
        meta = json.loads(rj.read_text())
        if meta.get("train_yaml") != "train.yaml" or meta["split_file"] not in UNIT_OF_SPLIT:
            continue
        pf = rj.parent / "predictions.csv"
        if not pf.is_file():
            warnings.warn(f"{rj.parent.name}: no predictions.csv, skipped")
            continue
        key = (meta["split_file"], meta.get("fold_id"))
        if key not in priors:
            sp = pd.read_csv(Path(paths.splits_dir) / meta["split_file"])
            if key[1] not in (None, ""):
                sp = sp[sp["fold_id"].astype(str) == str(key[1])]
            tr = sp[sp.split_role == "train"].merge(cov, on=["subject_id", "site"])
            tr = tr.dropna(subset=["age"])
            priors[key] = CovariatePrior.fit(tr["age"], tr["susp"], tr["mut"])
        prior = priors[key]
        pred = pd.read_csv(pf)
        parts = {}
        for role in ("val", "test"):
            t = pred[pred.role == role].merge(cov[["subject_id", "age", "susp"]],
                                              on="subject_id", how="left")
            if t["age"].isna().any():
                raise ValueError(f"{rj.parent.name}/{role}: {int(t['age'].isna().sum())} cases without age")
            t = t.assign(z=np.log(np.clip(t["alpha"], _EPS, None)) - np.log(np.clip(t["beta"], _EPS, None)),
                         y=t["y_true"].astype(float))
            t["pi_cov"] = prior.predict(t["age"], t["susp"])
            t["pi_age"] = prior.predict(t["age"])
            parts[role] = t[["subject_id", "z", "y", "age", "susp", "pi_cov", "pi_age"]].reset_index(drop=True)
        out.append(RunData(rj.parent.name, UNIT_OF_SPLIT[meta["split_file"]], int(meta["seed"]),
                           prior, parts["val"], parts["test"]))
    return out


# --------------------------------------------------- evaluation 1: full test set

METHODS_FULL = {
    "a_raw": "補正なし",
    "b0_ratio_age": "比方式（年齢のみ）",
    "b_ratio": "比方式（年齢+疑い）",
    "c_om_cov": "出力整合（年齢+疑い）",
    "d_om_age": "出力整合（年齢のみ）",
    "e_om_oracle": "出力整合（oracle 係留）",
    "f_label_mle": "全例ラベル切片",
}


def offsets_full(rd: RunData, test_susp=None) -> dict:
    """Intercepts a-f for one run on the whole test set. `test_susp` overrides the
    target-side suspicion answers (evaluation 3); the source side stays clean."""
    te, va = rd.test, rd.val
    pi_te = rd.prior.predict(te["age"], te["susp"] if test_susp is None else test_susp)
    z = te["z"].to_numpy()
    out, status = {"a_raw": 0.0}, {}
    out["b0_ratio_age"] = ratio_offset(te["pi_age"].mean(), va["pi_age"].mean())
    out["b_ratio"] = ratio_offset(pi_te.mean(), va["pi_cov"].mean())
    for k, target in (("c_om_cov", pi_te.mean()), ("d_om_age", te["pi_age"].mean()),
                      ("e_om_oracle", te["y"].mean())):
        out[k], status[k] = solve_offset(z, float(target))
    out["f_label_mle"] = label_intercept_mle(z, te["y"])
    return {"offsets": out, "status": status, "pi_cov_test": float(pi_te.mean()),
            "pi_cov_val": float(va["pi_cov"].mean()), "pi_age_test": float(te["pi_age"].mean()),
            "pi_true": float(te["y"].mean()), "pi_val_obs": float(va["y"].mean())}


def evaluate_full(runs: list[RunData]) -> pd.DataFrame:
    """Evaluation 1: one row per (run, method)."""
    rows = []
    for rd in runs:
        o = offsets_full(rd)
        for k, b in o["offsets"].items():
            m = calib_metrics(rd.test["z"], rd.test["y"], b)
            rows.append({"run": rd.run, "unit": rd.unit, "seed": rd.seed, "method": k,
                         "b": b, "status": o["status"].get(k, "ok"), **m,
                         **{x: o[x] for x in ("pi_cov_test", "pi_cov_val", "pi_age_test",
                                              "pi_true", "pi_val_obs")}})
    return pd.DataFrame(rows)


# ------------------------------------------------------ evaluation 2: N cases

def n_curve(runs: list[RunData], ns=NS, reps: int = 50, seed: int = 0,
            min_eval: int = 20) -> pd.DataFrame:
    """Evaluation 2: the first N cases of a new site, drawn per subject.

    From the SAME N cases: their covariates give pi_cov (-> ratio and output
    matching), their predictions give the output-matching measurement, and — for
    the labelled comparison — their labels give the intercept MLE. Every method is
    then scored on the remaining cases. N must leave `min_eval` cases (as in
    scripts/sample_size_curve.py). A labelled draw with a single class is not
    scored (its MLE is infinite), as in sample_size_curve; the count is kept.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rd in runs:
        te = rd.test
        z, y = te["z"].to_numpy(), te["y"].to_numpy()
        pic = te["pi_cov"].to_numpy()
        pi_src = float(rd.val["pi_cov"].mean())
        b_full = label_intercept_mle(z, y)                        # reference: all labels
        b_prior_oracle = ratio_offset(float(y.mean()), float(rd.val["y"].mean()))
        for n in ns:
            if len(y) < n + min_eval:
                continue
            for r in range(reps):
                idx = rng.permutation(len(y))
                f, ev = idx[:n], idx[n:]
                zf = z[f]
                pi_n = float(pic[f].mean())
                b_om, st = solve_offset(zf, pi_n)
                b_ratio = ratio_offset(pi_n, pi_src)
                single = len(np.unique(y[f])) < 2
                b_lab, st_lab = (np.nan, "single_class") if single else solve_offset(zf, float(y[f].mean()))
                base = {"run": rd.run, "unit": rd.unit, "seed": rd.seed, "n": n, "rep": r}
                raw_ece = calib_metrics(z[ev], y[ev])["ece"]
                for method, b, s in (("raw", 0.0, "ok"), ("om", b_om, st), ("ratio", b_ratio, "ok"),
                                     ("label", b_lab, st_lab), ("full_label", b_full, "ok"),
                                     ("prior_oracle", b_prior_oracle, "ok")):
                    ece = np.nan if np.isnan(b) else calib_metrics(z[ev], y[ev], b)["ece"]
                    rows.append({**base, "method": method, "b": b, "status": s,
                                 "ece": ece, "ece_raw": raw_ece})
    d = pd.DataFrame(rows)
    d["harm"] = np.where(d["ece"].isna(), np.nan, (d["ece"] > d["ece_raw"]).astype(float))
    return d


def summarise_n(d: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    """Per unit x N x method: ECE mean and 95% interval, sd of b, P(harm).

    p_harm is the specified reading (any increase over uncorrected ECE). Where
    there is no offset to recover (in-distribution) it sits near 50% whatever the
    method, because b-hat is pure noise around 0 and ECE on a finite remainder
    moves either way; p_harm_tol (worse by more than `tol`) separates noise from
    real damage and is reported alongside, not instead."""
    def agg(g):
        e = g["ece"].dropna()
        return pd.Series({
            "reps": len(g), "scored": len(e),
            "ece_mean": e.mean(), "ece_lo": e.quantile(.025), "ece_hi": e.quantile(.975),
            "b_mean": g["b"].mean(), "b_sd": g["b"].std(ddof=1),
            "p_harm": g["harm"].mean(),
            "p_harm_tol": (g["ece"] - g["ece_raw"] > tol).where(g["ece"].notna()).mean(),
            "clipped": int(g["status"].str.startswith("clip").sum()),
            "single_class": int((g["status"] == "single_class").sum())})
    return d.groupby(["unit", "n", "method"]).apply(agg, include_groups=False).reset_index()


def min_safe_n(summary: pd.DataFrame, method: str, threshold: float = 0.10,
               col: str = "p_harm") -> dict:
    """Smallest N with P(harm) < threshold, per unit (None if no N qualifies).

    Monotone reading: the smallest N from which every larger tested N also stays
    below the threshold, so a lucky dip at small N does not set the rule.
    """
    out = {}
    for unit, g in summary[summary.method == method].sort_values("n").groupby("unit"):
        ok = (g[col] < threshold).to_numpy()
        ns = g["n"].to_numpy()
        n_safe = None
        for i in range(len(ns)):
            if ok[i:].all():
                n_safe = int(ns[i])
                break
        out[unit] = n_safe
    return out


def label_equivalent(summary: pd.DataFrame, unit: str, method: str = "om") -> dict:
    """For each N of `method`, the number of LABELLED cases with the same mean ECE,
    by log-linear interpolation on the labelled curve. '<5' / '>200' outside."""
    s = summary[summary.unit == unit]
    lab = s[s.method == "label"].sort_values("n")
    ln, le = np.log(lab["n"].to_numpy(float)), lab["ece_mean"].to_numpy(float)
    out = {}
    for _, r in s[s.method == method].sort_values("n").iterrows():
        e = r["ece_mean"]
        if e >= le[0]:
            out[int(r["n"])] = f"<{int(lab['n'].iloc[0])}"
            continue
        if e <= le.min():
            out[int(r["n"])] = f">={int(lab['n'].iloc[int(np.argmin(le))])}"
            continue
        # first crossing along increasing N
        for i in range(len(le) - 1):
            if le[i] >= e >= le[i + 1]:
                t = (le[i] - e) / (le[i] - le[i + 1]) if le[i] != le[i + 1] else 0.0
                out[int(r["n"])] = f"{np.exp(ln[i] + t * (ln[i + 1] - ln[i])):.0f}"
                break
        else:
            out[int(r["n"])] = "n/a"
    return out


# ----------------------------------------------------- evaluation 3: robustness

def robustness(runs: list[RunData], flip: float = 0.15, missing=(0.0, .25, .5, 1.0),
               reps: int = 50, seed: int = 0) -> pd.DataFrame:
    """Evaluation 3 on the whole test set: target-side suspicion misread
    (marginal-preserving flips) and answered 'unknown' (-> age-only pi)."""
    rng = np.random.default_rng(seed)
    rows = []
    for rd in runs:
        z, y = rd.test["z"], rd.test["y"]
        susp = rd.test["susp"].to_numpy()
        conds = [("flip", flip, reps)] + \
                [("missing", m, 1 if m in (0.0, 1.0) else reps) for m in missing]
        for cond, level, k in conds:
            for r in range(k):
                if cond == "flip":
                    s = flip_balanced(susp, level, rng)
                elif cond == "missing":
                    s = mask_unknown(susp, level, rng)
                else:
                    s = susp
                o = offsets_full(rd, test_susp=s)
                for method in ("b_ratio", "c_om_cov"):
                    b = o["offsets"][method]
                    rows.append({"run": rd.run, "unit": rd.unit, "seed": rd.seed,
                                 "cond": cond, "level": level, "rep": r, "method": method,
                                 "b": b, "pi_cov_test": o["pi_cov_test"],
                                 "ece": calib_metrics(z, y, b)["ece"]})
    return pd.DataFrame(rows)
