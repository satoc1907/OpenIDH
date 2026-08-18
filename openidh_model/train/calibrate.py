"""Post-hoc temperature scaling (spec §10 事後較正).

Base rates move between splits — LOSO fold B goes from 11.7% mutant in train to
28.4% in test — so the fused probability is miscalibrated on the shifted test set
even when the ranking survives. Fit one scalar T on the inner-val split by
minimising NLL, apply it to test, and report ECE both raw and calibrated.

Temperature acts on the fused Beta's logit, ``log(alpha/beta) == logit(p_hat)``,
which is strictly increasing in p_hat. So AUC and AUPRC are invariant under it —
only the calibration metrics move. `check_ranking_invariant` asserts that.

The evidence mass S = alpha + beta is left untouched: temperature says nothing
about how much evidence there is, only about how it splits between the classes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ..eval.metrics import compute_metrics

_EPS = 1e-7


def to_logit(alpha, beta) -> np.ndarray:
    """Fused Beta -> binary logit. logit(alpha/(alpha+beta)) == log(alpha/beta)."""
    a = np.clip(np.asarray(alpha, dtype=np.float64), _EPS, None)
    b = np.clip(np.asarray(beta, dtype=np.float64), _EPS, None)
    return np.log(a) - np.log(b)


def apply_affine(alpha, beta, T: float = 1.0, offset: float = 0.0):
    """z -> z/T + offset, keeping S = alpha + beta fixed.

    T alone is temperature scaling; offset alone is a prior/base-rate correction;
    both together is Platt scaling. Every variant is monotone in z, so none of
    them can move AUC or AUPRC.
    """
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    S = a + b
    p = 1.0 / (1.0 + np.exp(-(to_logit(a, b) / float(T) + float(offset))))
    return p * S, S - p * S


def apply_temperature(alpha, beta, T: float):
    """Scale the logit by 1/T, keeping S = alpha + beta fixed."""
    return apply_affine(alpha, beta, T=T)


def prior_offset(pi_source: float, pi_target: float) -> float:
    """log-odds shift from a source base rate to a target one.

    The correction the fold-B miscalibration actually calls for: inner-val is
    11.8% mutant, its test set 28.4%, and temperature — having no intercept —
    cannot express that. Using the *observed* test prevalence makes this an
    oracle: it is an upper bound on what base-rate correction can buy, not a
    deployable method.
    """
    def _lo(p):
        p = float(np.clip(p, _EPS, 1 - _EPS))
        return np.log(p / (1 - p))
    return float(_lo(pi_target) - _lo(pi_source))


def fit_temperature(alpha, beta, y, bounds: tuple[float, float] = (0.02, 50.0)) -> float:
    """T >= 0 minimising inner-val NLL. T > 1 softens, T < 1 sharpens."""
    z = to_logit(alpha, beta)
    yv = np.asarray(y, dtype=np.float64)

    def nll(logT: float) -> float:
        p = np.clip(1.0 / (1.0 + np.exp(-z / np.exp(logT))), _EPS, 1 - _EPS)
        return float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))

    r = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(r.x))


def em_prior_shift(probs, pi_source: float, n_iter: int = 100, tol: float = 1e-6) -> dict:
    """Saerens-Latinne-Decaestecker EM: estimate the target prevalence from
    UNLABELLED predictions, then re-weight by the prior ratio.

    The zero-shot counterpart to prior_offset(): same one-parameter correction,
    but the target prevalence is read off the shape of the predicted-probability
    distribution instead of from labels. Returns the estimate, the corrected
    probabilities, and enough diagnostics to tell convergence from a stall.

    The re-weighting is exactly a logit offset of prior_offset(pi_source, pi_hat),
    so it inherits the ranking invariance.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1 - _EPS)
    pi_s = float(np.clip(pi_source, _EPS, 1 - _EPS))
    pi, trace = pi_s, []

    def _adjust(pi_cur):
        r, r0 = pi_cur / pi_s, (1 - pi_cur) / (1 - pi_s)
        return (r * p) / (r * p + r0 * (1 - p))

    converged, used = False, n_iter
    for i in range(n_iter):
        pi_new = float(_adjust(pi).mean())
        trace.append(pi_new)
        if abs(pi_new - pi) < tol:
            pi, converged, used = pi_new, True, i + 1
            break
        pi = pi_new
    return {
        "pi_hat": pi,
        "p_adj": _adjust(pi),                       # recomputed at the converged pi
        "offset": prior_offset(pi_s, pi),
        "n_iter": used, "converged": converged,
        # a late oscillation shows up here even when the tolerance is met
        "tail_range": float(np.ptp(trace[-5:])) if len(trace) >= 2 else 0.0,
    }


def calibrated_metrics(alpha, beta, y, T: float = 1.0, offset: float = 0.0) -> dict:
    a, b = apply_affine(alpha, beta, T=T, offset=offset)
    return compute_metrics(a, b, y)


def check_ranking_invariant(alpha, beta, y, T: float = 1.0, offset: float = 0.0,
                            tol: float = 1e-9) -> dict:
    """The map is monotone, so AUC/AUPRC must not move. Returns the deltas."""
    raw = compute_metrics(alpha, beta, y)
    cal = calibrated_metrics(alpha, beta, y, T=T, offset=offset)
    out = {k: cal[k] - raw[k] for k in ("auc", "auprc")}
    bad = {k: v for k, v in out.items() if not np.isnan(v) and abs(v) > tol}
    if bad:
        raise AssertionError(f"temperature scaling changed the ranking metrics: {bad}")
    return out
