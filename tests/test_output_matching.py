"""Output matching (openidh_model/eval/output_matching.py). No torch, no files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openidh_model.eval import output_matching as om  # noqa: E402


@pytest.fixture
def zy():
    rng = np.random.default_rng(0)
    z = rng.normal(-1.0, 2.0, size=400)
    y = (rng.random(400) < expit(z + 0.8)).astype(float)   # true intercept +0.8
    return z, y


def test_solve_offset_hits_target(zy):
    z, _ = zy
    for target in (0.05, 0.2, 0.5, 0.9):
        b, st = om.solve_offset(z, target)
        assert st == "ok" and abs(expit(z + b).mean() - target) < 1e-10


def test_oracle_anchor_equals_label_mle(zy):
    """Anchored at the observed prevalence, output matching IS the intercept score
    equation — it must reproduce the independently minimised NLL."""
    z, y = zy
    b_om, _ = om.solve_offset(z, y.mean())
    assert abs(b_om - om.label_intercept_mle(z, y)) < 1e-6


def test_unreachable_target_is_clipped_with_warning():
    z = np.full(50, -30.0)                      # even b=+10 leaves p ~ 0
    with pytest.warns(UserWarning):
        b, st = om.solve_offset(z, 0.3)
    assert (b, st) == (om.BRACKET[1], "clip_hi")


def test_constant_offset_leaves_auc(zy):
    z, y = zy
    assert om.calib_metrics(z, y, 1.3)["auc"] == om.calib_metrics(z, y, 0.0)["auc"]


def test_flip_balanced_keeps_marginal():
    rng = np.random.default_rng(1)
    s = (rng.random(20000) < 0.68).astype(float)
    s[:100] = np.nan
    f = om.flip_balanced(s, 0.15, rng)
    assert np.isnan(f[:100]).all()
    assert abs(np.nanmean(f) - np.nanmean(s)) < 0.01
    assert 0.12 < np.nanmean(f[100:] != s[100:]) < 0.25


def test_mask_unknown_fraction():
    rng = np.random.default_rng(2)
    s = np.ones(200)
    assert np.isnan(om.mask_unknown(s, 0.25, rng)).sum() == 50


def test_prior_falls_back_to_age_when_unknown():
    rng = np.random.default_rng(3)
    age = rng.uniform(20, 80, 500)
    susp = (rng.random(500) < 0.5).astype(float)
    y = (rng.random(500) < expit(-0.05 * (age - 50) - 1.5 * susp)).astype(int)
    p = om.CovariatePrior.fit(age, susp, y)
    s = susp.copy(); s[:10] = np.nan
    got = p.predict(age, s)
    assert np.allclose(got[:10], p.predict(age[:10]))
    assert not np.allclose(got[10:], p.predict(age[10:]))
