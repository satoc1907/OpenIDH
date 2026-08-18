"""Invariant tests (spec §12). Stage 0 subset.

Runnable now: parameter count, no train/test leakage across splits.
Fusion / KL / volume-scaling tests are added in Stage A when fusion.py and the
losses exist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from openidh_model.utils.paths import load_paths  # noqa: E402

PATHS = load_paths(_ROOT / "configs" / "paths.local.yaml")
EXPECTED_PARAMS = 44_224_396


@pytest.fixture(scope="module")
def model():
    from openidh_model.models.openidh import OpenIDH
    return OpenIDH(weights_dir=PATHS.weights_dir)


def test_param_count(model):
    assert model.num_parameters() == EXPECTED_PARAMS


def test_forward_shapes_and_positive(model):
    import torch
    from openidh_model.models.openidh import IMG_MODALITIES
    model.eval()
    imgs = {m: torch.randn(2, 3, 224, 224) for m in IMG_MODALITIES}
    with torch.no_grad():
        ev = model.raw_evidence(imgs, torch.randn(2, 2))
    assert set(ev) == set(IMG_MODALITIES) | {"unified", "tabular"}
    for v in ev.values():
        assert tuple(v.shape) == (2, 2)
        assert (v > 0).all()


@pytest.mark.parametrize(
    "fname",
    ["splits_loso_foldA.csv", "splits_loso_foldB.csv",
     "splits_vendor_philips.csv", "splits_field.csv", "splits_random_5fold.csv"],
)
def test_no_train_test_leak(fname):
    df = pd.read_csv(PATHS.splits_dir / fname)
    for fid, sf in df.groupby("fold_id"):
        train_val = set(sf[sf.split_role.isin(["train", "val"])].subject_id)
        test = set(sf[sf.split_role == "test"].subject_id)
        assert train_val & test == set(), f"{fname} fold {fid}: subject leak"


def test_no_idh_na_in_splits():
    for fname in ["splits_loso_foldA.csv", "splits_random_5fold.csv"]:
        df = pd.read_csv(PATHS.splits_dir / fname)
        assert not (df.idh == "NA").any()
        assert set(df.idh.unique()) <= {"Mut", "WT"}


# ── §12: KL regularizer (closed form) ──────────────────────────────────────
import math  # noqa: E402
import torch  # noqa: E402
from openidh_model.losses.evidential import loss_reg  # noqa: E402
from openidh_model.models.fusion import fuse, volume_scale  # noqa: E402


def _reg_scalar(e0, y=1.0):
    return float(loss_reg(torch.tensor([0.0]), torch.tensor([float(e0)]), torch.tensor([y]))[0])


def test_kl_reference_values():
    assert abs(_reg_scalar(1) - 0.1931472) < 1e-6
    assert abs(_reg_scalar(4) - 0.8094379) < 1e-6
    assert abs(_reg_scalar(9) - 1.4025851) < 1e-6


def test_kl_matches_closed_form():
    """Binary simple form log(w)-(w-1)/w equals the general digamma KL (a=1,b=w)."""
    for e0 in [0.0, 1.0, 4.0, 9.0, 100.0]:
        b = torch.tensor(e0 + 1.0, dtype=torch.float64)
        one = torch.tensor(1.0, dtype=torch.float64)
        full = (
            torch.lgamma(one + b) - torch.lgamma(one) - torch.lgamma(b)
            + (one - 1.0) * (torch.digamma(one) - torch.digamma(one + b))
            + (b - 1.0) * (torch.digamma(b) - torch.digamma(one + b))
        )
        simple = math.log(e0 + 1.0) - e0 / (e0 + 1.0)
        assert abs(float(full) - simple) < 1e-9


# ── §12: fusion invariants ─────────────────────────────────────────────────
_NAMES = ["T1", "T2", "FLAIR", "T1c", "unified", "tabular"]


def _ev(val=1.0, B=1):
    return {n: torch.full((B, 2), float(val)) for n in _NAMES}


def _active(flags, B=1):
    return {n: torch.tensor([flags[n]] * B) for n in _NAMES}


def test_all_missing_gives_uniform():
    a, be, _ = fuse(_ev(), _active({n: False for n in _NAMES}), torch.tensor([50_000.0]))
    assert torch.allclose(a, torch.ones_like(a))
    assert torch.allclose(be, torch.ones_like(be))


def test_missing_reduces_S():
    vol = torch.tensor([50_000.0])  # s(V)=1
    full = {n: True for n in _NAMES}
    drop1 = {**full, "T1": False}
    drop3 = {**full, "T1": False, "T2": False, "FLAIR": False}
    S = lambda flags: sum(fuse(_ev(), _active(flags), vol)[:2])
    assert float(S(full)) > float(S(drop1)) > float(S(drop3))


def test_small_volume_reduces_S():
    flags = {n: True for n in _NAMES}
    S_big = sum(fuse(_ev(), _active(flags), torch.tensor([100_000.0]))[:2])
    S_small = sum(fuse(_ev(), _active(flags), torch.tensor([100.0]))[:2])
    assert float(S_small) < float(S_big)
    assert abs(float(volume_scale(torch.tensor(100.0))) - 0.1) < 1e-4


def test_volume_not_in_forward_inputs():
    """Volume must reach the model only via fuse(), never as a classifier input."""
    import inspect
    from openidh_model.models.openidh import OpenIDH
    params = inspect.signature(OpenIDH.raw_evidence).parameters
    assert "volume" not in params  # raw evidence cannot see volume
    # and fuse actually moves S with volume (so the mechanism is live)
    flags = {n: True for n in _NAMES}
    S_a = sum(fuse(_ev(), _active(flags), torch.tensor([100.0]))[:2])
    S_b = sum(fuse(_ev(), _active(flags), torch.tensor([100_000.0]))[:2])
    assert not torch.allclose(S_a, S_b)


# ── post-hoc temperature scaling (spec §10) ──────────────────────────────────
def _beta_from_logit(z, S=40.0):
    import numpy as np
    p = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))
    return p * S, S - p * S


def test_temperature_preserves_ranking():
    """Temperature is monotone in p_hat, so AUC/AUPRC must not move (spec §10)."""
    import numpy as np
    from openidh_model.train.calibrate import check_ranking_invariant
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400).astype(float)
    a, b = _beta_from_logit(rng.normal(0, 2, 400) + 1.5 * y)
    for T in (0.25, 1.0, 3.7):
        d = check_ranking_invariant(a, b, y, T)   # raises if AUC/AUPRC move
        assert all(abs(v) < 1e-9 for v in d.values())


def test_temperature_recovers_known_scaling():
    """Logits inflated by a known factor should be undone by fitting T on them."""
    import numpy as np
    from openidh_model.train.calibrate import apply_temperature, fit_temperature
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-z))).astype(float)
    a, b = _beta_from_logit(z * 3.0)              # over-confident by 3x
    T = fit_temperature(a, b, y)
    assert 2.4 < T < 3.6, T
    from openidh_model.eval.metrics import compute_metrics
    assert compute_metrics(*apply_temperature(a, b, T), y)["ece"] < compute_metrics(a, b, y)["ece"]


def test_temperature_keeps_evidence_mass():
    """S = alpha + beta carries evidence, not class split — temperature must not touch it."""
    import numpy as np
    from openidh_model.train.calibrate import apply_temperature
    a, b = _beta_from_logit([-2.0, 0.0, 1.3], S=55.0)
    a2, b2 = apply_temperature(a, b, 2.5)
    assert np.allclose(np.asarray(a) + np.asarray(b), a2 + b2)


def test_prior_offset_matches_log_odds_ratio():
    """§10's fold-B shift: 11.8% inner-val -> 28.4% test is a +1.087 logit offset."""
    from openidh_model.train.calibrate import prior_offset
    assert prior_offset(0.118, 0.284) == pytest.approx(1.0868, abs=1e-3)
    assert prior_offset(0.3, 0.3) == pytest.approx(0.0, abs=1e-12)
    assert prior_offset(0.284, 0.118) == pytest.approx(-prior_offset(0.118, 0.284), abs=1e-12)


def test_prior_offset_shifts_the_logit_exactly():
    """The contract is additive on the logit; the mean probability only follows
    approximately, because sigmoid is nonlinear (Jensen)."""
    import numpy as np
    from openidh_model.train.calibrate import apply_affine, prior_offset, to_logit
    rng = np.random.default_rng(3)
    a, b = _beta_from_logit(rng.normal(-2.0, 1.0, 20000))
    pi_src = float(np.mean(a / (a + b)))
    off = prior_offset(pi_src, 0.284)
    a2, b2 = apply_affine(a, b, offset=off)
    assert np.allclose(to_logit(a2, b2), to_logit(a, b) + off, atol=1e-9)
    assert pi_src < float(np.mean(a2 / (a2 + b2))) < 0.284 + 0.03


def test_prior_offset_also_preserves_ranking():
    import numpy as np
    from openidh_model.train.calibrate import check_ranking_invariant
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300).astype(float)
    a, b = _beta_from_logit(rng.normal(0, 2, 300) + 1.2 * y)
    d = check_ranking_invariant(a, b, y, offset=1.108)
    assert all(abs(v) < 1e-9 for v in d.values())


# ── EM prior shift (spec §10, zero-shot variant) ─────────────────────────────
def _label_shift_scores(rng, n, pi_source, pi_target, mu=2.0):
    """Textbook label shift: p(x|y) fixed, prevalence moves, model still assumes
    pi_source. The posterior it emits is then exactly what EM is built to correct."""
    y = (rng.random(n) < pi_target).astype(float)
    x = rng.normal(mu * y, 1.0)
    z = np.log(pi_source / (1 - pi_source)) + mu * x - mu ** 2 / 2
    return 1 / (1 + np.exp(-z)), y


def test_em_recovers_prevalence_under_true_label_shift():
    """EM's assumption is p(x|y) fixed, p(y) moved. Under that, it must find p(y)."""
    from openidh_model.train.calibrate import em_prior_shift
    rng = np.random.default_rng(7)
    p, _ = _label_shift_scores(rng, 40000, pi_source=0.12, pi_target=0.30)
    out = em_prior_shift(p, pi_source=0.12)
    assert out["converged"]
    assert out["pi_hat"] == pytest.approx(0.30, abs=0.03), out["pi_hat"]


def test_em_is_a_noop_when_nothing_shifted():
    from openidh_model.train.calibrate import em_prior_shift
    rng = np.random.default_rng(8)
    p, _ = _label_shift_scores(rng, 40000, pi_source=0.18, pi_target=0.18)
    out = em_prior_shift(p, pi_source=0.18)
    assert out["pi_hat"] == pytest.approx(0.18, abs=0.02)
    assert abs(out["offset"]) < 0.15


def test_em_correction_preserves_ranking():
    from openidh_model.train.calibrate import check_ranking_invariant, em_prior_shift
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 500).astype(float)
    a, b = _beta_from_logit(rng.normal(-1.0, 1.8, 500) + 1.4 * y)
    out = em_prior_shift(a / (a + b), pi_source=0.15)
    d = check_ranking_invariant(a, b, y, offset=out["offset"])
    assert all(abs(v) < 1e-9 for v in d.values())
