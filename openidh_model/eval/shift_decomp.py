"""Feature-space decomposition of a domain shift (experiment A).

For one encoder, the mean CLS feature moves between inner-val (source sites)
and test (shifted site) by

    delta = mean(feat_test) - mean(feat_val)            in R^384

The evidence head Linear(384->2) only "sees" a 2-d subspace of that space,
span{W1, W0} (W1 = e1 row, W0 = e0 row). Orthonormalise it into

    u_p = (W1 - W0) / ||W1 - W0||        discrimination direction: moves p_hat
    u_s = orth(W1 + W0; u_p), normalised confidence direction: moves S

and split delta into

    delta_p    = <delta, u_p>            criterion shift (an intercept change)
    delta_s    = <delta, u_s>            evidence-mass shift (a confidence change)
    delta_null = || delta - delta_p u_p - delta_s u_s ||
                                         movement this head cannot see

In 384 dimensions the orthogonal residual is trivially the largest of the three,
so raw norms decide nothing. Everything is reported as an effect size instead:
delta_p / sigma_p with sigma_p the s.d. of the val features along u_p (same for
s), and delta_null against a noise floor — the residual norm of a "null delta"
between two random halves of the val set, repeated `n_boot` times.

Functional translation (first order, softplus gradient ignored): the encoder's
contribution to the fused logit moves by ~<delta, W1 - W0>, and its evidence
mass by ~<delta, W1 + W0>. Because the features are dumped, the exact
non-linear shift of the head output is also computed and reported next to it.
"""
from __future__ import annotations

import numpy as np

from .features import E0_ROW, E1_ROW, ENCODERS, evidence_from_features

__all__ = ["ENCODERS", "functional_basis", "project", "shift_decomposition",
           "kitagawa_split", "age_grade_strata"]


def functional_basis(W: np.ndarray) -> dict:
    """Orthonormal (u_p, u_s) spanning the head's functional plane."""
    W = np.asarray(W, dtype=np.float64)
    W1, W0 = W[E1_ROW], W[E0_ROW]
    d = W1 - W0
    u_p = d / np.linalg.norm(d)
    v = W1 + W0
    v_perp = v - (v @ u_p) * u_p
    u_s = v_perp / np.linalg.norm(v_perp)
    return {"u_p": u_p, "u_s": u_s, "w_diff": d, "w_sum": v,
            # how oblique the head's two rows are: 0 = orthogonal, 1 = parallel
            "cos_sum_diff": float((v @ d) / (np.linalg.norm(v) * np.linalg.norm(d)))}


def project(delta: np.ndarray, basis: dict) -> dict:
    delta = np.asarray(delta, dtype=np.float64)
    dp, ds = float(delta @ basis["u_p"]), float(delta @ basis["u_s"])
    resid = delta - dp * basis["u_p"] - ds * basis["u_s"]
    return {"delta_p": dp, "delta_s": ds, "delta_null": float(np.linalg.norm(resid)),
            "delta_norm": float(np.linalg.norm(delta))}


def _null_draws(feat_val: np.ndarray, basis: dict, sigma_p: float, sigma_s: float,
                n_boot: int, rng: np.random.Generator) -> dict:
    """Noise floor: split val into two random halves, take the mean difference,
    decompose it. Returns the three components over `n_boot` draws (abs for the
    two signed ones)."""
    n = feat_val.shape[0]
    half = n // 2
    dp, ds, dn = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.permutation(n)
        d0 = feat_val[idx[:half]].mean(0) - feat_val[idx[half:2 * half]].mean(0)
        pr = project(d0, basis)
        dp[i] = abs(pr["delta_p"]) / sigma_p
        ds[i] = abs(pr["delta_s"]) / sigma_s
        dn[i] = pr["delta_null"]
    return {"d_p": dp, "d_s": ds, "null": dn, "half": half}


def shift_decomposition(feat_val: np.ndarray, feat_test: np.ndarray, W: np.ndarray,
                        b: np.ndarray | None = None, n_boot: int = 100,
                        seed: int = 0) -> dict:
    """Full decomposition for one encoder of one run. All inputs float32 arrays.

    Returns effect sizes (d_p, d_s), the residual and its noise floor, the
    first-order functional translation (dz_lin, dS_lin) and, when `b` is given,
    the exact head-output shift computed from the features themselves.
    """
    fv = np.asarray(feat_val, dtype=np.float64)
    ft = np.asarray(feat_test, dtype=np.float64)
    basis = functional_basis(W)
    delta = ft.mean(0) - fv.mean(0)
    pr = project(delta, basis)

    sigma_p = float((fv @ basis["u_p"]).std(ddof=1))
    sigma_s = float((fv @ basis["u_s"]).std(ddof=1))
    rng = np.random.default_rng(seed)
    null = _null_draws(fv, basis, sigma_p, sigma_s, n_boot, rng)

    n_val, n_test, half = fv.shape[0], ft.shape[0], null["half"]
    # The half-split floor is measured at variance scale 2/half, while the real
    # delta is at 1/n_val + 1/n_test. Under iid noise the residual norm scales
    # with the square root of that, so this factor makes the floor size-matched.
    # It is < 1 (val halves are smaller than either set): the raw spec floor is
    # the conservative one, the matched floor the fair one. Both are reported.
    match = float(np.sqrt((1.0 / n_val + 1.0 / n_test) / (2.0 / half)))

    d_p, d_s = pr["delta_p"] / sigma_p, pr["delta_s"] / sigma_s
    floor = {k: {"median": float(np.median(v)), "p95": float(np.quantile(v, 0.95))}
             for k, v in null.items() if k != "half"}
    # analytic s.e. of a standardised mean difference (iid): same for d_p and d_s
    se_d = float(np.sqrt(1.0 / n_val + 1.0 / n_test))

    out = {
        "n_val": int(n_val), "n_test": int(n_test), "n_boot": int(n_boot),
        "delta_p": pr["delta_p"], "delta_s": pr["delta_s"],
        "delta_null": pr["delta_null"], "delta_norm": pr["delta_norm"],
        "sigma_p": sigma_p, "sigma_s": sigma_s,
        "d_p": float(d_p), "d_s": float(d_s), "se_d": se_d,
        "floor": floor, "floor_match": match,
        # ratios to the noise floor (median of the null draws)
        "d_p_over_floor": float(abs(d_p) / floor["d_p"]["median"]),
        "d_s_over_floor": float(abs(d_s) / floor["d_s"]["median"]),
        "null_over_floor": float(pr["delta_null"] / floor["null"]["median"]),
        "null_over_floor_matched": float(pr["delta_null"] / (floor["null"]["median"] * match)),
        # first-order functional translation
        "dz_lin": float(delta @ basis["w_diff"]),     # ~ shift of the head's logit
        "dS_lin": float(delta @ basis["w_sum"]),      # ~ shift of the head's evidence mass
        "cos_sum_diff": basis["cos_sum_diff"],
        "w_diff_norm": float(np.linalg.norm(basis["w_diff"])),
        "w_sum_norm": float(np.linalg.norm(basis["w_sum"])),
    }
    if b is not None:  # exact (softplus) head output, from the very same features
        ev_v = evidence_from_features(feat_val, np.asarray(W, np.float32), np.asarray(b, np.float32))
        ev_t = evidence_from_features(feat_test, np.asarray(W, np.float32), np.asarray(b, np.float32))
        S_v, S_t = ev_v.sum(1), ev_t.sum(1)
        out["S_val_mean"] = float(S_v.mean()); out["S_test_mean"] = float(S_t.mean())
        out["dS_exact"] = float(S_t.mean() - S_v.mean())
        out["dS_exact_rel"] = float(S_t.mean() / S_v.mean() - 1.0)
        lz = lambda ev: np.log(ev[:, E1_ROW]) - np.log(ev[:, E0_ROW])  # noqa: E731
        out["dz_exact"] = float(lz(ev_t).mean() - lz(ev_v).mean())
    return out


def kitagawa_split(feat_val: np.ndarray, feat_test: np.ndarray,
                   strata_val: np.ndarray, strata_test: np.ndarray,
                   min_n: int = 3) -> dict:
    """Split delta into a within-stratum part and a composition part.

    With pi_k the test share of stratum k and mu the stratum means,

        delta = sum_k pi_test_k (mu_test_k - mu_val_k)         "within"  (imaging)
              + sum_k (pi_test_k - pi_val_k) mu_val_k          "composition" (case mix)

    which is exact when every test stratum also has val cases. Strata with fewer
    than `min_n` cases on either side are dropped from the within term (their
    test share is reported as `uncovered`); composition is then delta - within,
    which also absorbs what the uncovered strata would have contributed.
    """
    fv = np.asarray(feat_val, dtype=np.float64); ft = np.asarray(feat_test, dtype=np.float64)
    sv = np.asarray(strata_val); st = np.asarray(strata_test)
    delta = ft.mean(0) - fv.mean(0)
    keys = sorted(set(sv.tolist()) | set(st.tolist()))
    within = np.zeros_like(delta); covered = 0.0; per = {}
    for k in keys:
        mv, mt = sv == k, st == k
        pi_t, pi_v = mt.mean(), mv.mean()
        if mv.sum() >= min_n and mt.sum() >= min_n:
            dk = ft[mt].mean(0) - fv[mv].mean(0)
            within += pi_t * dk
            covered += pi_t
            per[k] = {"n_val": int(mv.sum()), "n_test": int(mt.sum()),
                      "pi_val": float(pi_v), "pi_test": float(pi_t), "delta_k": dk}
        else:
            per[k] = {"n_val": int(mv.sum()), "n_test": int(mt.sum()),
                      "pi_val": float(pi_v), "pi_test": float(pi_t), "delta_k": None}
    if covered > 0:
        within = within / covered   # renormalise the test weights over covered strata
    return {"delta": delta, "within": within, "composition": delta - within,
            "covered": float(covered), "strata": per}


def age_grade_strata(age: np.ndarray, grade: np.ndarray, age_cuts: tuple[float, float]) -> np.ndarray:
    """(grade binary x age tertile) stratum labels, e.g. 'hi/a1'. NA grade -> 'na';
    NA age -> 'a?' (kept as its own bin rather than dropped)."""
    age = np.asarray(age, dtype=float); grade = np.asarray(grade, dtype=float)
    g = np.where(np.isnan(grade), "na", np.where(grade == 4, "hi", "lo"))
    a = np.full(age.shape, "a?", dtype=object)
    a[age <= age_cuts[0]] = "a1"
    a[(age > age_cuts[0]) & (age <= age_cuts[1])] = "a2"
    a[age > age_cuts[1]] = "a3"
    return np.array([f"{x}/{y}" for x, y in zip(g, a)])
