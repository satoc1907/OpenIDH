"""Shift decomposition (experiment A) — algebraic invariants on synthetic data.

No checkpoint or image data needed: these guard the basis construction, the
sign convention, the Kitagawa identity and the noise-floor scaling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openidh_model.eval.features import E0_ROW, E1_ROW, evidence_from_features  # noqa: E402
from openidh_model.eval.shift_decomp import (  # noqa: E402
    age_grade_strata, functional_basis, kitagawa_split, project, shift_decomposition,
)

D = 384


@pytest.fixture
def W():
    rng = np.random.default_rng(1)
    return rng.normal(size=(2, D)).astype(np.float32)


def test_basis_orthonormal_and_spans_head(W):
    b = functional_basis(W)
    assert abs(np.linalg.norm(b["u_p"]) - 1) < 1e-12
    assert abs(np.linalg.norm(b["u_s"]) - 1) < 1e-12
    assert abs(b["u_p"] @ b["u_s"]) < 1e-12
    # both head rows lie in span{u_p, u_s}: their residual off the plane is 0
    for row in W.astype(np.float64):
        resid = row - (row @ b["u_p"]) * b["u_p"] - (row @ b["u_s"]) * b["u_s"]
        assert np.linalg.norm(resid) < 1e-9


def test_sign_convention_moves_p_hat_up(W):
    """A shift along +u_p must raise e1 relative to e0 (p_hat up), and a shift
    along +u_s must raise e1 + e0 (S up), when applied to the actual head."""
    b = functional_basis(W)
    f0 = np.zeros((1, D), np.float32)
    bias = np.zeros(2, np.float32)
    e_base = evidence_from_features(f0, W, bias)[0]
    e_p = evidence_from_features((0.5 * b["u_p"]).astype(np.float32)[None], W, bias)[0]
    e_s = evidence_from_features((0.5 * b["u_s"]).astype(np.float32)[None], W, bias)[0]
    lz = lambda e: np.log(e[E1_ROW]) - np.log(e[E0_ROW])  # noqa: E731
    assert lz(e_p) > lz(e_base)
    assert e_s.sum() > e_base.sum()


def test_decomposition_recovers_planted_components(W):
    b = functional_basis(W)
    rng = np.random.default_rng(0)
    n_val, n_test = 200, 400
    sigma = 0.3
    feat_val = rng.normal(scale=sigma, size=(n_val, D))
    # plant: +2 sigma along u_p, -1 sigma along u_s, and a big orthogonal move
    orth = rng.normal(size=D)
    orth -= (orth @ b["u_p"]) * b["u_p"] + (orth @ b["u_s"]) * b["u_s"]
    orth *= 5.0 / np.linalg.norm(orth)
    shift = 2 * sigma * b["u_p"] - 1 * sigma * b["u_s"] + orth
    feat_test = rng.normal(scale=sigma, size=(n_test, D)) + shift

    r = shift_decomposition(feat_val, feat_test, W, n_boot=50)
    assert abs(r["d_p"] - 2.0) < 0.35
    assert abs(r["d_s"] + 1.0) < 0.35
    # residual = planted 5.0 + noise (~ sigma * sqrt(D*(1/n_val+1/n_test)) ~ 0.5)
    assert 4.5 < r["delta_null"] < 6.0
    assert r["null_over_floor"] > 3          # far above the noise floor
    assert 0 < r["floor_match"] < 1
    # the linear translation is the in-plane projection re-expressed in W units
    assert abs(r["dz_lin"] - r["delta_p"] * r["w_diff_norm"]) < 1e-9
    v_par = (W.astype(float).sum(0)) @ b["u_p"]
    v_perp = np.linalg.norm(W.astype(float).sum(0) - v_par * b["u_p"])
    assert abs(r["dS_lin"] - (r["delta_p"] * v_par + r["delta_s"] * v_perp)) < 1e-9


def test_null_shift_sits_on_the_floor(W):
    """Two samples from the same distribution: every component near its floor."""
    rng = np.random.default_rng(3)
    fv = rng.normal(size=(200, D)); ft = rng.normal(size=(400, D))
    r = shift_decomposition(fv, ft, W, n_boot=100)
    assert abs(r["d_p"]) < 3 * r["se_d"]
    assert abs(r["d_s"]) < 3 * r["se_d"]
    # the raw half-split floor is conservative (larger than the matched one)
    assert r["null_over_floor"] < 1.0
    assert 0.7 < r["null_over_floor_matched"] < 1.3


def test_kitagawa_identity_and_pure_composition():
    rng = np.random.default_rng(5)
    mu = {"a": rng.normal(size=D), "b": rng.normal(size=D) + 3}
    def draw(n_a, n_b):
        f = np.concatenate([mu["a"] + 0.01 * rng.normal(size=(n_a, D)),
                            mu["b"] + 0.01 * rng.normal(size=(n_b, D))])
        s = np.array(["a"] * n_a + ["b"] * n_b)
        return f, s
    fv, sv = draw(100, 100)
    ft, st = draw(50, 150)          # same within-stratum means, different mix
    r = kitagawa_split(fv, ft, sv, st)
    assert r["covered"] == 1.0
    np.testing.assert_allclose(r["within"] + r["composition"], r["delta"], atol=1e-12)
    assert np.linalg.norm(r["within"]) < 0.1 * np.linalg.norm(r["composition"])
    # and the reverse: same mix, shifted stratum means -> all "within"
    ft2 = fv + 1.0
    r2 = kitagawa_split(fv, ft2, sv, sv)
    assert np.linalg.norm(r2["composition"]) < 1e-9
    np.testing.assert_allclose(r2["within"], np.ones(D), atol=1e-9)


def test_strata_labels():
    age = np.array([30, 50, 70, np.nan])
    grade = np.array([2, 4, np.nan, 4])
    lab = age_grade_strata(age, grade, (45, 60))
    assert lab.tolist() == ["lo/a1", "hi/a2", "na/a3", "hi/a?"]


def test_project_residual_is_orthogonal(W):
    b = functional_basis(W)
    d = np.random.default_rng(9).normal(size=D)
    pr = project(d, b)
    resid = d - pr["delta_p"] * b["u_p"] - pr["delta_s"] * b["u_s"]
    assert abs(resid @ b["u_p"]) < 1e-9 and abs(resid @ b["u_s"]) < 1e-9
    assert abs(np.linalg.norm(resid) - pr["delta_null"]) < 1e-12
