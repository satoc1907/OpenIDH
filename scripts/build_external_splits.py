"""Evaluation lists for the external cohorts (EGD, Japan) — inference only.

Reads each cohort's OpenIDH-format table from Paths.external_metadata_dir, keeps
the cases that can be scored, and writes one split CSV per cohort in the same
format as the frozen splits (subject_id, site, idh, split_role, fold_id), with
every row `split_role = test`. Extra columns are for subgroup analysis only:

  all4        1 when T1, T2, FLAIR and T1c are all present
  cohort_site the cohort's own site code (Japan: 10 institutions; EGD: blank)

Selection (counts expected from the data-management reports, 2026-11-05):
  EGD   qc_passed, IDH Mut/WT, primary seg                    -> 467 (Mut 155 / WT 312)
  Japan qc_passed, IDH Mut/WT, primary seg, no flag in
        JAPAN_CHECK_FLAGS                                      -> 620 (Mut 175 / WT 445)
A cohort whose count differs from the expected one is reported, not silently used.

The CSVs carry labels, so they are written outside the repo (default:
<external_metadata_dir>/splits/) and never committed.

    uv run python scripts/build_external_splits.py            # OPENIDH_PATHS=configs/paths.hpc.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openidh_model.data.metadata import EXTERNAL_COHORTS, resolve_subject_dir  # noqa: E402
from openidh_model.utils.paths import load_paths  # noqa: E402

SEQS = ["T1", "T2", "FLAIR", "T1c"]
# label_flag values that mean "check this case before using its label"
# (Japan report §3-2). clinical_from_2021_only is NOT one: 77 of the 620 are 2021-only.
JAPAN_CHECK_FLAGS = {
    "no_clinical", "dup_id_in_master2021", "patho_FALSE_2019", "age_under18",
    "excluded_in_master2021", "on_hold_in_master2021", "idh_uncertain",
}
EXPECTED = {"EGD": (467, 155, 312), "Japan": (620, 175, 445)}


def _true(col: pd.Series) -> pd.Series:
    return col.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def _flags(v) -> set[str]:
    return {f.strip() for f in str(v).split(";") if f.strip() and f.strip().lower() != "nan"}


def select(df: pd.DataFrame, site: str) -> tuple[pd.DataFrame, dict]:
    """(kept rows, {reason: n dropped}). Reasons are applied in order."""
    drop: dict[str, int] = {}
    keep = pd.Series(True, index=df.index)
    rules = [
        ("qc_passed is not true", _true(df["qc_passed"])),
        ("IDH not Mut/WT", df["idh"].astype(str).str.strip().isin({"Mut", "WT"})),
        ("no primary seg", _true(df["has_seg"])),
    ]
    if site == "Japan":
        rules.append(("label_flag needs checking",
                      ~df["label_flag"].map(lambda v: bool(_flags(v) & JAPAN_CHECK_FLAGS))))
    for reason, ok in rules:
        drop[reason] = int((keep & ~ok).sum())
        keep &= ok
    return df[keep], drop


def to_split(kept: pd.DataFrame, site: str) -> pd.DataFrame:
    all4 = pd.concat([_true(kept[f"has_{m}"]) for m in SEQS], axis=1).all(axis=1)
    return pd.DataFrame({
        "subject_id": kept["subject_id"].astype(str).str.strip(),
        "site": site,
        "idh": kept["idh"].astype(str).str.strip(),
        "split_role": "test",
        "fold_id": f"ext_{site.lower()}",
        "all4": all4.astype(int).values,
        "cohort_site": kept["site"].astype(str).values if "site" in kept else "",
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-file", default=None, help="defaults to $OPENIDH_PATHS")
    ap.add_argument("--out-dir", default=None, help="default: <external_metadata_dir>/splits")
    ap.add_argument("--cohorts", nargs="*", default=list(EXTERNAL_COHORTS))
    args = ap.parse_args()

    paths = load_paths(args.paths_file)
    if paths.external_metadata_dir is None:
        sys.exit("external_metadata_dir is not set in the paths YAML")
    out_dir = Path(args.out_dir or paths.external_metadata_dir / "splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    mismatch = []
    for site in args.cohorts:
        table = paths.external_metadata_dir / EXTERNAL_COHORTS[site][1]
        df = pd.read_csv(table, dtype={"subject_id": str}, keep_default_na=False)
        kept, drop = select(df, site)
        split = to_split(kept, site)

        no_img = [s for s in split.subject_id
                  if not (resolve_subject_dir(paths.data_dir, site, s) / "volumes"
                          / "tumor_seg.nii.gz").is_file()]
        split = split[~split.subject_id.isin(no_img)]

        f = out_dir / f"splits_ext_{site.lower()}.csv"
        split.to_csv(f, index=False)
        n, mut = len(split), int((split.idh == "Mut").sum())
        print(f"== {site}: {table.name} {len(df)} rows -> {n} cases "
              f"(Mut {mut} / WT {n - mut}, all 4 sequences {int(split.all4.sum())})")
        for reason, k in drop.items():
            print(f"   dropped {k:4d}  {reason}")
        print(f"   dropped {len(no_img):4d}  no volumes/tumor_seg.nii.gz on disk"
              + (f": {no_img[:10]}" if no_img else ""))
        print(f"   -> {f}")
        if (n, mut, n - mut) != EXPECTED[site]:
            mismatch.append(f"{site}: got {n} (Mut {mut} / WT {n - mut}), "
                            f"report says {EXPECTED[site][0]} "
                            f"(Mut {EXPECTED[site][1]} / WT {EXPECTED[site][2]})")
    if mismatch:
        print("\n!! count differs from the data-management report — check before evaluating:")
        for m in mismatch:
            print("   " + m)
        sys.exit(1)
    print("\ncounts match the reports.")


if __name__ == "__main__":
    main()
