"""GliomaDataset — one item per split row (spec §3).

Returns per-modality (3,224,224) image tensors (the model concatenates them into
the 12ch unified input), an availability mask, the whole-tumor volume (kept OUT
of the image path — spec §1.2 #3), tabular [age, sex], and the IDH label.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .metadata import lookup_age_sex, resolve_subject_dir
from .transforms import IMAGE_SIZE, select_slices, slices_to_tensor

MODALITIES = ["T1", "T2", "FLAIR", "T1c"]


class GliomaDataset(Dataset):
    def __init__(
        self,
        split_csv,
        paths,
        role: str | None = "train",
        fold_id=None,
        image_size: int = IMAGE_SIZE,
        age_mean: float | None = None,
        age_std: float | None = None,
        skip_missing_dirs: bool = True,
    ):
        df = pd.read_csv(split_csv)
        # random_5fold.csv holds 5 folds in one file; filter to one before role.
        if fold_id is not None:
            df = df[df["fold_id"].astype(str) == str(fold_id)]
        if role is not None:
            df = df[df["split_role"] == role]
        self.paths = paths
        self.image_size = image_size
        self.age_mean = age_mean
        self.age_std = age_std

        rows, missing = [], []
        for _, r in df.iterrows():
            d = resolve_subject_dir(paths.data_dir, r["site"], r["subject_id"])
            if skip_missing_dirs and not (d / "volumes").is_dir():
                missing.append(r["subject_id"])
                continue
            rows.append(r)
        self.rows = rows
        self.missing = missing  # subjects whose preprocessed images are absent

    def __len__(self):
        return len(self.rows)

    def _load_seg(self, vol_dir: Path) -> np.ndarray:
        return np.asarray(nib.load(vol_dir / "tumor_seg.nii.gz").dataobj)

    def __getitem__(self, idx):
        r = self.rows[idx]
        vol_dir = resolve_subject_dir(self.paths.data_dir, r["site"], r["subject_id"]) / "volumes"

        seg = self._load_seg(vol_dir)
        z_idx = select_slices(seg)
        volume = float((seg > 0).sum())  # whole tumor (labels 1+2+4)

        images, mask = {}, {}
        for m in MODALITIES:
            f = vol_dir / f"{m}.nii.gz"
            if f.is_file():
                arr = np.asarray(nib.load(f).dataobj)
                images[m] = slices_to_tensor(arr, z_idx, self.image_size)
                mask[m] = True
            else:  # missing modality -> zero-fill + mask False (spec §3.5)
                images[m] = torch.zeros(3, self.image_size, self.image_size)
                mask[m] = False

        age, sex = lookup_age_sex(self.paths.metadata_dir, r["site"], r["subject_id"])
        if self.age_mean is not None and self.age_std:
            age = (age - self.age_mean) / self.age_std
        tabular = torch.tensor([age, sex], dtype=torch.float32)

        label = 1 if str(r["idh"]).strip() == "Mut" else 0
        return {
            "images": images,
            "mask": mask,
            "volume": torch.tensor(volume, dtype=torch.float32),
            "tabular": tabular,
            "label": torch.tensor(label, dtype=torch.float32),
            "subject_id": r["subject_id"],
            "site": r["site"],
        }
