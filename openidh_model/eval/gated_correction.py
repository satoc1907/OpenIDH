"""Significance-gated intercept correction (notebook 07).

Evaluation 2 of output matching (06) left the operating rule unit-dependent: at
N=20 the correction is almost never harmful on LOSO-B but harmful in most draws
in-distribution, on field and on vendor — and a portal cannot know in advance
which kind of site it is facing. The gate lets the estimate decide for itself:

    apply b-hat only if |b-hat| > k * SE(b-hat), else b = 0

SE comes from a bootstrap INSIDE the N cases (resample with replacement,
recompute the anchor, re-solve b), so the decision uses nothing a deployment
would not have: the N cases' covariates and predictions (plus their labels for
the labelled comparator), and source-side constants fixed before deployment
(pi-hat on the val covariates, for the ratio method). Never the test labels,
never the remaining cases. `gate_decision` takes only those arrays, which is
what makes the leak check a matter of its signature; a test additionally
perturbs the remainder and asserts the decision does not move.

The same gate is put on all three estimators — output matching, the ratio
method, the labelled intercept — so the comparison stays fair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .output_matching import BRACKET, NS, RunData, calib_metrics, logit, ratio_offset, solve_offset

KS = (1.0, 1.5, 2.0, 2.5)
METHODS = ("om", "ratio", "label")
N_BOOT = 200
MAX_CLIP = 0.10


def solve_offsets_batch(Z, targets, bracket=BRACKET, iters: int = 50):
    """Vectorised solve_offset: row r of Z gets b_r with mean(sigmoid(Z_r + b_r))
    == targets_r. Bisection, 50 halvings of a width-20 bracket (~2e-14).
    Returns (b, clipped) where clipped marks rows whose target is out of reach —
    their b is pinned to the bracket end, exactly as solve_offset does."""
    Z = np.asarray(Z, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    lo_b, hi_b = bracket
    f = lambda b: expit(Z + b[:, None]).mean(1) - t
    lo = np.full(len(t), lo_b)
    hi = np.full(len(t), hi_b)
    clip_lo = f(lo) >= 0
    clip_hi = f(hi) <= 0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = f(mid) < 0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    b = 0.5 * (lo + hi)
    b = np.where(clip_lo, lo_b, np.where(clip_hi, hi_b, b))
    return b, clip_lo | clip_hi


def estimate(method: str, zf, pif, yf, pi_src: float):
    """Point estimate on the N cases. Returns (b, determinable)."""
    if method == "om":
        b, st = solve_offset(zf, float(np.mean(pif)))
        return b, st == "ok"
    if method == "ratio":
        return ratio_offset(float(np.mean(pif)), pi_src), True
    if method == "label":
        if len(np.unique(yf)) < 2:           # MLE at infinity: nothing to gate on
            return np.nan, False
        b, st = solve_offset(zf, float(np.mean(yf)))
        return b, st == "ok"
    raise ValueError(method)


def bootstrap_b(method: str, zf, pif, yf, pi_src: float, rng, n_boot: int = N_BOOT):
    """b re-solved on n_boot resamples (with replacement) of the N cases.
    Returns (b_boot, clipped_mask). Clipped resamples are kept, not dropped."""
    n = len(zf)
    idx = rng.integers(0, n, size=(n_boot, n))
    if method == "om":
        return solve_offsets_batch(zf[idx], pif[idx].mean(1))
    if method == "ratio":
        p = np.clip(pif[idx].mean(1), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)) - logit(pi_src), np.zeros(n_boot, bool)
    if method == "label":
        return solve_offsets_batch(zf[idx], yf[idx].mean(1))   # 0 or 1 -> clipped
    raise ValueError(method)


def gate_decision(method: str, zf, pif, yf, pi_src: float, rng, ks=KS,
                  n_boot: int = N_BOOT, max_clip: float = MAX_CLIP) -> dict:
    """Everything the gate needs, from the N cases only.

    Inputs are the N cases' predictions (zf), covariate prevalences (pif), their
    labels (yf — read by the labelled method only) and the source constant
    pi_src. Undeterminable (-> no correction) when the point estimate is not
    finite / clipped, or more than `max_clip` of the bootstrap re-solves clip.
    """
    zf, pif, yf = (np.asarray(x, dtype=np.float64) for x in (zf, pif, yf))
    b, ok = estimate(method, zf, pif, yf, pi_src)
    bb, clipped = bootstrap_b(method, zf, pif, yf, pi_src, rng, n_boot)
    clip_rate = float(clipped.mean())
    se = float(np.std(bb, ddof=1))
    undet = (not ok) or clip_rate > max_clip
    fired = {k: bool((not undet) and abs(b) > k * se) for k in ks}
    return {"b": b, "se": se, "clip_rate": clip_rate, "undeterminable": undet, "fired": fired}


def gated_curve(runs: list[RunData], ns=NS, reps: int = 50, seed: int = 0, ks=KS,
                n_boot: int = N_BOOT, min_eval: int = 20) -> pd.DataFrame:
    """Evaluation 2 of notebook 06 with the gate added.

    The N-case draws reproduce 06 exactly (same generator, same order, same skip
    rule), so the ungated columns must equal 06's n_curve. The bootstrap has its
    own generator. One row per (run, N, rep, method) with ungated and per-k ECE.
    """
    rng_draw = np.random.default_rng(seed)
    rng_boot = np.random.default_rng(seed + 1)
    rows = []
    for rd in runs:
        te = rd.test
        z, y, pic = te["z"].to_numpy(), te["y"].to_numpy(), te["pi_cov"].to_numpy()
        pi_src = float(rd.val["pi_cov"].mean())
        for n in ns:
            if len(y) < n + min_eval:
                continue
            for r in range(reps):
                idx = rng_draw.permutation(len(y))
                f, ev = idx[:n], idx[n:]
                # ---- gate: N cases only. Nothing indexed by `ev` enters here.
                decisions = {m: gate_decision(m, z[f], pic[f], y[f], pi_src, rng_boot, ks, n_boot)
                             for m in METHODS}
                # ---- scoring: the remainder, after every decision is fixed
                raw = calib_metrics(z[ev], y[ev])["ece"]
                for m, d in decisions.items():
                    ece_hat = np.nan if np.isnan(d["b"]) else calib_metrics(z[ev], y[ev], d["b"])["ece"]
                    row = {"run": rd.run, "unit": rd.unit, "seed": rd.seed, "n": n, "rep": r,
                           "method": m, "b": d["b"], "se": d["se"], "clip_rate": d["clip_rate"],
                           "undeterminable": d["undeterminable"], "ece_raw": raw,
                           "ece_ungated": ece_hat}
                    for k in ks:
                        row[f"fired_{k}"] = d["fired"][k]
                        row[f"ece_{k}"] = ece_hat if d["fired"][k] else raw
                    rows.append(row)
    return pd.DataFrame(rows)


def summarise_gated(d: pd.DataFrame, ks=KS, tol: float = 0.01) -> pd.DataFrame:
    """Per unit x N x method x gate ('none' or k): fire rate, ECE mean,
    P(harm) and P(harm > tol), undeterminable rate. For 'none' the ungated ECE
    is used and draws without a finite estimate (single-class labelled draws)
    are not scored, as in 06."""
    out = []
    for (unit, n, m), g in d.groupby(["unit", "n", "method"], observed=True):
        raw = g["ece_raw"]
        for gate in ("none",) + tuple(ks):
            e = g["ece_ungated"] if gate == "none" else g[f"ece_{gate}"]
            ok = e.notna()
            diff = (e - raw)[ok]
            out.append({
                "unit": unit, "n": n, "method": m, "gate": str(gate),
                "reps": len(g), "scored": int(ok.sum()),
                "fire_rate": float(np.nan if gate == "none" else g[f"fired_{gate}"].mean()),
                "undeterminable": float(g["undeterminable"].mean()),
                "clip_rate_mean": float(g["clip_rate"].mean()),
                "ece_mean": float(e[ok].mean()), "ece_raw": float(raw[ok].mean()),
                "p_harm": float((diff > 0).mean()), "p_harm_tol": float((diff > tol).mean()),
                "se_median": float(g["se"].median()),
            })
    return pd.DataFrame(out)


def choose_k(summary: pd.DataFrame, method: str = "om", ks=KS, n_from: int = 20,
             threshold: float = 0.10, loss_unit: str = "LOSO-B"):
    """The pre-registered rule. Among k where every unit has P(harm>0.01) <
    threshold at every tested N >= n_from, pick the one with the smallest mean
    ECE loss against the ungated estimator on `loss_unit` over those N; ties go
    to the smaller k. Returns (k or None, table of candidates)."""
    s = summary[summary.method == method]
    ung = s[s.gate == "none"].set_index(["unit", "n"])["ece_mean"]
    rows = []
    for k in ks:
        g = s[(s.gate == str(k)) & (s.n >= n_from)]
        worst = g.groupby("unit", observed=True)["p_harm_tol"].max()
        lb = g[g.unit == loss_unit].set_index(["unit", "n"])["ece_mean"]
        rows.append({"k": k, "passes": bool((worst < threshold).all()),
                     "worst_p_harm_tol": float(worst.max()),
                     "worst_unit": str(worst.idxmax()),
                     f"{loss_unit}_loss": float((lb - ung.reindex(lb.index)).mean())})
    t = pd.DataFrame(rows)
    ok = t[t.passes]
    if ok.empty:
        return None, t
    best = ok.sort_values([f"{loss_unit}_loss", "k"]).iloc[0]
    return float(best["k"]), t


def n_min(summary: pd.DataFrame, k, method: str = "om", threshold: float = 0.10):
    """Smallest N from which every unit keeps P(harm>0.01) < threshold at every
    larger tested N (None if not even the largest N qualifies)."""
    s = summary[(summary.method == method) & (summary.gate == str(k))]
    worst = s.groupby("n")["p_harm_tol"].max().sort_index()
    ns, ok = worst.index.to_numpy(), (worst < threshold).to_numpy()
    for i in range(len(ns)):
        if ok[i:].all():
            return int(ns[i])
    return None
