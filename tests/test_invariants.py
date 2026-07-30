"""Invariant tests (spec §12). Stage 0 subset.

Runnable now: parameter count, no train/test leakage across splits.
Fusion / KL / volume-scaling tests are added in Stage A when fusion.py and the
losses exist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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
