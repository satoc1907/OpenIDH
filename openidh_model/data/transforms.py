"""Slice selection and tensor building (spec §3.2, §3.3).

Augmentation (bias field, noise, intensity/gamma — spec §8) is a Stage-A-later
concern; its parameters are tuned on inner-val. This module holds the
deterministic geometry: VOI slice choice and 256->224 resize.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_SIZE = 224


def select_slices(seg: np.ndarray) -> list[int]:
    """Indices of the max-tumor-area axial slice and its two neighbors (±1)."""
    area = (seg > 0).sum(axis=(0, 1))          # tumor area per z
    z = int(area.argmax())
    z = int(np.clip(z, 1, seg.shape[2] - 2))   # avoid running off the ends
    return [z - 1, z, z + 1]


def slices_to_tensor(vol: np.ndarray, z_idx: list[int], size: int = IMAGE_SIZE) -> torch.Tensor:
    """Stack the 3 chosen slices as channels and resize to (3, size, size)."""
    chans = np.stack([vol[:, :, z] for z in z_idx], axis=0).astype(np.float32)  # (3, H, W)
    t = torch.from_numpy(chans).unsqueeze(0)                                    # (1, 3, H, W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0)                                                          # (3, size, size)
