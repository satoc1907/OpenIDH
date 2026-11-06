"""External cohorts (EGD, Japan): path/metadata lookup, case selection, and the
missing-age path in the forward pass. Synthetic tables only — no real data."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import build_external_splits as bes  # noqa: E402
from openidh_model.data.metadata import lookup_age_sex, resolve_subject_dir  # noqa: E402

COLS = ["subject_id", "age", "sex", "idh", "has_T1", "has_T2", "has_FLAIR", "has_T1c",
        "has_seg", "qc_passed"]


def _table(rows, extra=None):
    df = pd.DataFrame(rows, columns=COLS + (extra or []))
    return df.astype(str)


def test_resolve_subject_dir_external(tmp_path):
    assert resolve_subject_dir(tmp_path, "EGD", "MR_EGD-0001") == tmp_path / "egd" / "MR_EGD-0001"
    assert resolve_subject_dir(tmp_path, "Japan", "2007") == tmp_path / "japan_lps" / "2007"


def test_lookup_age_sex_external_keeps_ids_and_gaps(tmp_path):
    _table([["MR_EGD-0001", "77", "M", "WT", 1, 1, 1, 1, 1, "true"],
            ["MR_EGD-0002", "", "F", "Mut", 1, 1, 1, 1, 1, "true"],
            ["MR_EGD-0003", "NA", "", "WT", 1, 1, 1, 1, 1, "true"]]
           ).to_csv(tmp_path / "EGD_metadata_openidh.csv", index=False)
    assert lookup_age_sex(tmp_path, "EGD", "MR_EGD-0001") == (77.0, 0.0)
    age, sex = lookup_age_sex(tmp_path, "EGD", "MR_EGD-0002")
    assert math.isnan(age) and sex == 1.0
    assert all(math.isnan(v) for v in lookup_age_sex(tmp_path, "EGD", "MR_EGD-0003"))
    assert all(math.isnan(v) for v in lookup_age_sex(tmp_path, "EGD", "absent"))

    _table([["0012", "50", "F", "WT", 1, 1, 1, 1, 1, "true"]]
           ).to_csv(tmp_path / "Japan_metadata_openidh_who2021.csv", index=False)
    assert lookup_age_sex(tmp_path, "Japan", "0012") == (50.0, 1.0)  # leading zero kept


def test_select_japan_drops_flags_but_not_2021_only():
    df = _table([
        ["1001", "60", "M", "Mut", 1, 1, 1, 1, 1, "true", "clinical_from_2021_only"],  # kept
        ["1002", "60", "M", "WT", 1, 1, 1, 0, 1, "true", ""],                          # kept, not all4
        ["1003", "60", "M", "WT", 1, 1, 1, 1, 1, "true", "dup_id_in_master2021"],
        ["1007", "60", "M", "WT", 1, 1, 1, 1, 1, "true", "patho_FALSE_2019"],         # kept
        ["1004", "17", "F", "WT", 1, 1, 1, 1, 1, "true", "clinical_from_2021_only;age_under18"],
        ["1005", "60", "M", "NA", 1, 1, 1, 1, 1, "true", ""],
        ["1006", "60", "M", "WT", 1, 1, 1, 1, 0, "false", ""],
    ], extra=["label_flag"])
    kept, drop = bes.select(df, "Japan")
    assert list(kept.subject_id) == ["1001", "1002", "1007"]
    assert drop == {"qc_passed is not true": 1, "IDH not Mut/WT": 1, "no primary seg": 0,
                    "label_flag needs checking": 2}
    split = bes.to_split(kept, "Japan")
    assert list(split.all4) == [1, 0, 1]
    assert set(split.split_role) == {"test"} and set(split.site) == {"Japan"}


def test_missing_age_drops_only_the_tabular_head():
    torch = pytest.importorskip("torch")
    from openidh_model.train.loop import _forward

    class Stub(torch.nn.Module):
        def raw_evidence(self, images, tabular):
            assert not torch.isnan(tabular).any()      # NaN never reaches the head
            ev = {m: torch.ones(len(tabular), 2) for m in ["T1", "T2", "FLAIR", "T1c", "unified"]}
            ev["tabular"] = torch.full((len(tabular), 2), 3.0)
            return ev

    B = 2
    batch = {
        "images": {m: torch.zeros(B, 3, 4, 4) for m in ["T1", "T2", "FLAIR", "T1c"]},
        "mask": {m: torch.ones(B, dtype=torch.bool) for m in ["T1", "T2", "FLAIR", "T1c"]},
        "volume": torch.full((B,), 1e6),
        "tabular": torch.tensor([[0.1, 0.0], [float("nan"), 1.0]]),
        "label": torch.zeros(B),
    }
    cfg = {"train": {"modality_dropout_p": 0.0}, "fusion": {"v_ref": 10_000, "gamma": 0.5}}
    _, active, alpha, beta, _, _ = _forward(Stub(), batch, cfg, False, "cpu", torch.Generator())
    assert active["tabular"].tolist() == [True, False]
    assert alpha.tolist() == [1 + 5 + 3, 1 + 5]     # 5 image heads x 1, + tabular 3 when present
    assert not torch.isnan(beta).any()
