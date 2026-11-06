"""Join split subject_ids to clinical metadata (age, sex) and to image dirs.

IDH comes from the split CSVs (already Mut/WT). Age/sex come from each site's
metadata table, which uses different column names and ID formats (spec §2 notes).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd


# External-validation cohorts (inference only, never trained on): site ->
# (image folder under data_dir, metadata table). Both tables use the OpenIDH
# column format (subject_id, age, sex, idh, ...), are not in git, and are read
# from Paths.external_metadata_dir. Their subject_id is the image folder name.
EXTERNAL_COHORTS = {
    "EGD": ("egd", "EGD_metadata_openidh.csv"),
    "Japan": ("japan_lps", "Japan_metadata_openidh_who2021.csv"),
}


def resolve_subject_dir(data_dir, site: str, subject_id: str) -> Path:
    """Map a split subject_id to its preprocessed volume directory."""
    data_dir = Path(data_dir)
    if site in EXTERNAL_COHORTS:
        return data_dir / EXTERNAL_COHORTS[site][0] / str(subject_id)  # MR_EGD-0001 / 2007
    if site == "UTSW":
        return data_dir / "utsw-glioma" / subject_id            # BT0001
    if site == "UPenn":
        return data_dir / "upenn-gbm" / subject_id              # UPENN-GBM-00001_11
    if site == "UCSF":
        n = int(re.search(r"(\d+)", subject_id).group(1))
        return data_dir / "ucsf-pdgm" / f"UCSF-PDGM-{n:04d}_nifti"
    raise ValueError(f"unknown site: {site}")


def _sex_to_int(v) -> float:
    s = str(v).strip().upper()
    if s.startswith("M"):
        return 0.0
    if s.startswith("F"):
        return 1.0
    return float("nan")


@lru_cache(maxsize=8)
def _load_clinical(metadata_dir: str) -> dict:
    """Return {site: {key: (age, sex)}} where key is the join key per site."""
    md = Path(metadata_dir)
    out: dict = {"UCSF": {}, "UTSW": {}, "UPenn": {}}

    u = pd.read_csv(md / "UCSF-PDGM-metadata_v5.csv")
    for _, r in u.iterrows():
        n = int(re.search(r"(\d+)", str(r["ID"])).group(1))
        out["UCSF"][n] = (float(r["Age at MRI"]), _sex_to_int(r["Sex"]))

    t = pd.read_csv(md / "UTSW_Glioma_Metadata.tsv", sep="\t")
    for _, r in t.iterrows():
        out["UTSW"][str(r["Subject ID"])] = (
            pd.to_numeric(r["Age at Imaging"], errors="coerce"),
            _sex_to_int(r["Sex at birth"]),
        )

    c = pd.read_csv(md / "UPENN-GBM_clinical_info_v2.1.csv")
    for _, r in c.iterrows():
        out["UPenn"][str(r["ID"])] = (
            pd.to_numeric(r["Age_at_scan_years"], errors="coerce"),
            _sex_to_int(r["Gender"]),
        )
    return out


@lru_cache(maxsize=8)
def _load_external(metadata_dir: str, site: str) -> dict:
    """{subject_id: (age, sex)} from an external cohort's OpenIDH-format table."""
    t = pd.read_csv(Path(metadata_dir) / EXTERNAL_COHORTS[site][1],
                    dtype={"subject_id": str}, keep_default_na=False)
    return {r["subject_id"]: (float(pd.to_numeric(r["age"], errors="coerce")),
                              _sex_to_int(r["sex"]))
            for _, r in t.iterrows()}


def lookup_age_sex(metadata_dir, site: str, subject_id: str):
    """(age_years, sex_int) for a subject, or (nan, nan) if not found.

    For an external cohort, `metadata_dir` is Paths.external_metadata_dir.
    """
    if site in EXTERNAL_COHORTS:
        return _load_external(str(metadata_dir), site).get(
            str(subject_id), (float("nan"), float("nan")))
    table = _load_clinical(str(metadata_dir))
    if site == "UCSF":
        key = int(re.search(r"(\d+)", subject_id).group(1))
    else:
        key = str(subject_id)
    return table.get(site, {}).get(key, (float("nan"), float("nan")))


@lru_cache(maxsize=8)
def _load_grade(metadata_dir: str) -> dict:
    """{site: {key: grade}} — WHO grade as float, nan when unknown. UPenn carries
    no grade column: the cohort is UPENN-GBM (glioblastoma), so grade 4 by the
    dataset's definition rather than a value read from metadata."""
    md = Path(metadata_dir)
    out: dict = {"UCSF": {}, "UTSW": {}}
    u = pd.read_csv(md / "UCSF-PDGM-metadata_v5.csv")
    for _, r in u.iterrows():
        n = int(re.search(r"(\d+)", str(r["ID"])).group(1))
        out["UCSF"][n] = float(pd.to_numeric(r["WHO CNS Grade"], errors="coerce"))
    t = pd.read_csv(md / "UTSW_Glioma_Metadata.tsv", sep="\t")
    for _, r in t.iterrows():
        out["UTSW"][str(r["Subject ID"])] = float(pd.to_numeric(r["Tumor Grade"], errors="coerce"))
    return out


def lookup_grade(metadata_dir, site: str, subject_id: str) -> float:
    """WHO grade (2/3/4) for a subject; nan if unknown. UPenn -> 4.0 by cohort definition."""
    if site == "UPenn":
        return 4.0
    table = _load_grade(str(metadata_dir))
    key = int(re.search(r"(\d+)", subject_id).group(1)) if site == "UCSF" else str(subject_id)
    return table.get(site, {}).get(key, float("nan"))
