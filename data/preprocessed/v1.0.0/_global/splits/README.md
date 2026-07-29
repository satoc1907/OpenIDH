# Data splits — the single source of truth for all experiments

**Canonical generator: [`make_splits.ipynb`](./make_splits.ipynb)** (re-run top-to-bottom to regenerate
every CSV here). A Japanese-commented version, [`make_splits_ja.ipynb`](./make_splits_ja.ipynb), is
identical in logic and produces byte-identical CSVs. This is the citable artifact for the paper's Methods
section. **seed = 42** (recorded in every stochastic step). These CSVs are frozen: all downstream
experiments must read splits from here, never re-generate on the fly.

## Cohort definition (IDH-known only; NA fully excluded)

| site | n | Mut | WT | source | IDH rule |
|------|---|-----|----|--------|----------|
| UCSF  | 501 | 103 | 398 | `metadata/UCSF-PDGM-metadata_v5.csv` (all 501 sessions) | `wildtype`→WT, else→Mut (no NA present) |
| UTSW  | 619 | 176 | 443 | `utsw-glioma/_global/seg_source_inventory.csv` qc_passed==true (622) ∩ `metadata/UTSW_Glioma_Metadata.tsv` | `wild type`→WT, `mutated`→Mut; 3 IDH-NA dropped |
| UPenn | 515 | 16  | 499 | `metadata/UPENN-GBM_clinical_info_v2.1.csv` ∩ preprocessed images (611) | `Wildtype`→WT, `Mutated`→Mut; 96 `NOS/NEC` dropped as NA |
| **TOTAL** | **1635** | **295** | **1340** | | pos_rate = 18.0% |

## Patient-level grouping (no patient spans train & test)

- `patient_id` is the grouping key.
- UCSF: numeric ID normalized → `UCSF-<n>`. **495 patients / 501 sessions**; 6 patients have 2 sessions
  (baseline + follow-up): **391, 396, 409, 429, 431, 433**. Both sessions always land on the same side.
- UTSW: 619 patients / 619 sessions (no multi-session).
- UPenn: 515 patients / 515 sessions (base ID before `_11` timepoint; no multi-session).

## Splits produced

| file | design | test | train |
|------|--------|------|-------|
| `splits_loso_foldA.csv` | site-LOSO | UCSF | UTSW + UPenn |
| `splits_loso_foldB.csv` | site-LOSO | UTSW | UCSF + UPenn |
| `splits_vendor_philips.csv` | UTSW leave-one-vendor-out | UTSW Philips (140) | UTSW Siemens+GE+Hitachi+Toshiba+NotReported (479) |
| `splits_field.csv` | UTSW field-strength | UTSW 3T (195) | UTSW rest = 1.5T + minority + NotReported (424) |
| `splits_random_5fold.csv` | all-site random 5-fold | fold_id 0–4 (rotating) | remaining folds |

UPenn is **never** a test site (only 16 Mut → unstable positive-class evaluation).
Minority UTSW vendors / odd field strengths are placed on the **train** side (per spec).

## Inner validation (hyperparameter / early-stopping)

Each outer `train` is further split 80/20 into inner-`train` / `val` using
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, taking the first fold as `val`.
→ patient-grouped, stratified on `site×IDH`. The outer `test` is evaluated **once**.

## File schema

`subject_id, site, idh, split_role, fold_id`
- `split_role` ∈ {train, val, test}  (train = inner-train; val = inner-val)
- `fold_id`: `loso_A`/`loso_B` / `vendor_philips` / `field_3T` / `0..4` (random)

## Stratification / balance

- Outer random 5-fold: stratified on `site×IDH` → site ratio **and** Mut/WT ratio preserved per fold
  (every fold ≈ 18% positive).
- LOSO folds inherit each site's native base rate → **test positive rate differs by fold**
  (UCSF 20.6%, UTSW 28.4%): calibrate any decision threshold on inner-val, not a global constant.

## Reproduce

Run the notebook (from the repo root or any subdir — it auto-locates the repo root):

```
jupyter nbconvert --to notebook --execute --inplace \
  data/preprocessed/v1.0.0/_global/splits/make_splits.ipynb
```

Deterministic given seed=42 and identical input metadata (verified: regenerated CSVs are byte-identical
across runs). Integrity checks (subject overlap, patient leakage, NA contamination) are `assert`ed in the
final cell and must all pass.
