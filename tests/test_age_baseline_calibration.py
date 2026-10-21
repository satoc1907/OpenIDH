"""Calibration helpers of scripts/age_baseline_calibration.py. No torch, no files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import age_baseline_calibration as abc  # noqa: E402
from openidh_model.eval.metrics import _ece  # noqa: E402


@pytest.fixture
def z():
    return np.random.default_rng(0).normal(-1.0, 2.0, size=20000)


def test_intercept_recovers_known_offset(z):
    rng = np.random.default_rng(1)
    y = (rng.random(z.size) < expit(z + 0.8)).astype(float)   # score under-predicts by 0.8
    assert abc.calibration_intercept(expit(z), y) == pytest.approx(0.8, abs=0.08)


def test_slope_recovers_overconfidence(z):
    rng = np.random.default_rng(2)
    y = (rng.random(z.size) < expit(0.3 + 0.5 * z)).astype(float)   # overconfident score
    a, s = abc.calibration_slope(expit(z), y)
    assert s == pytest.approx(0.5, abs=0.05)
    assert a == pytest.approx(0.3, abs=0.08)


def test_calibrated_score_is_identity(z):
    rng = np.random.default_rng(3)
    p = expit(z)
    y = (rng.random(z.size) < p).astype(float)
    m = abc.score_metrics(p, y)
    assert m["cal_intercept"] == pytest.approx(0.0, abs=0.06)
    assert m["cal_slope"] == pytest.approx(1.0, abs=0.06)
    assert m["ece"] < 0.02


def test_reliability_bins_reproduce_ece(z):
    rng = np.random.default_rng(4)
    p = expit(z - 0.5)
    y = (rng.random(z.size) < expit(z)).astype(float)
    bins = abc.reliability(p, y)
    ece = sum(b["n"] / z.size * abs(b["frac_pos"] - b["mean_p"]) for b in bins)
    assert ece == pytest.approx(_ece(p, y), abs=1e-12)
    assert sum(b["n"] for b in bins) == z.size
