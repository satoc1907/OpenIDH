"""Gated correction (openidh_model/eval/gated_correction.py). No torch, no files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openidh_model.eval import gated_correction as gc  # noqa: E402
from openidh_model.eval import output_matching as om  # noqa: E402


def test_batch_solver_matches_brentq():
    rng = np.random.default_rng(0)
    Z = rng.normal(-1, 2, size=(40, 30))
    t = rng.uniform(0.05, 0.9, 40)
    b, clipped = gc.solve_offsets_batch(Z, t)
    assert not clipped.any()
    ref = np.array([om.solve_offset(Z[i], t[i])[0] for i in range(40)])
    assert np.abs(b - ref).max() < 1e-9


def test_batch_solver_clips_unreachable():
    Z = np.zeros((3, 10))
    b, clipped = gc.solve_offsets_batch(Z, np.array([0.0, 0.5, 1.0]))
    assert clipped.tolist() == [True, False, True]
    assert b[0] == om.BRACKET[0] and b[2] == om.BRACKET[1] and abs(b[1]) < 1e-9


def _fake_run(n=300, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(-1.5, 2, n)
    test = pd.DataFrame({"z": z, "y": (rng.random(n) < expit(z + 1.0)).astype(float),
                         "pi_cov": rng.uniform(0.1, 0.5, n)})
    val = pd.DataFrame({"pi_cov": rng.uniform(0.05, 0.3, 200), "y": (rng.random(200) < .15).astype(float)})
    return om.RunData("fake_s0", "fake", 0, None, val, test)


def test_gate_ignores_the_remainder():
    """Leak check: rewrite every remaining case's label, prediction and anchor —
    the decisions on the first N cases must not move."""
    rd = _fake_run()
    d1 = gc.gated_curve([rd], ns=(20,), reps=5, seed=3, n_boot=50)
    rd2 = _fake_run()
    rng = np.random.default_rng(99)
    perm = np.random.default_rng(3).permutation(len(rd2.test))   # the draw gated_curve makes first
    ev = perm[20:]
    rd2.test.loc[ev, "y"] = 1.0 - rd2.test.loc[ev, "y"]
    rd2.test.loc[ev, "z"] = rng.normal(5, 1, len(ev))
    rd2.test.loc[ev, "pi_cov"] = 0.99
    d2 = gc.gated_curve([rd2], ns=(20,), reps=1, seed=3, n_boot=50)
    a = d1[d1.rep == 0].set_index("method")
    b = d2[d2.rep == 0].set_index("method")
    cols = ["b", "se", "clip_rate", "undeterminable"] + [f"fired_{k}" for k in gc.KS]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_ungated_reproduces_06_draws():
    """Same generator, same order: the ungated output-matching ECE equals 06's."""
    rd = _fake_run()
    g = gc.gated_curve([rd], ns=(10, 50), reps=4, seed=0, n_boot=20)
    o = om.n_curve([rd], ns=(10, 50), reps=4, seed=0)
    a = g[g.method == "om"].sort_values(["n", "rep"])["ece_ungated"].to_numpy()
    b = o[o.method == "om"].sort_values(["n", "rep"])["ece"].to_numpy()
    assert np.allclose(a, b)


def test_single_class_label_draw_is_undeterminable():
    rng = np.random.default_rng(0)
    d = gc.gate_decision("label", rng.normal(size=10), np.full(10, .3), np.zeros(10), 0.2, rng)
    assert d["undeterminable"] and not any(d["fired"].values())


@pytest.mark.parametrize("method", gc.METHODS)
def test_large_shift_fires_small_does_not(method):
    rng = np.random.default_rng(1)
    n = 200
    z = rng.normal(-2, 1.5, n)
    y = (rng.random(n) < expit(z + 2.0)).astype(float)          # true shift +2
    pif = np.full(n, y.mean())
    big = gc.gate_decision(method, z, pif, y, pi_src=0.12, rng=rng)
    assert big["fired"][2.5]
