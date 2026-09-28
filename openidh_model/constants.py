"""Constants shared by the model and by the analysis code.

They live here, outside any module that imports torch, so the analysis path
(load an npz, decompose, tabulate) runs on a machine that cannot install torch
— the lock pins CUDA-only wheels, so a Mac has none.
"""
from __future__ import annotations

# Fixed channel order for the unified 12ch input (spec §3.3 — must not change).
IMG_MODALITIES = ["T1", "T2", "FLAIR", "T1c"]
