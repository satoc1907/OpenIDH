"""Modality dropout (spec §7, §4.2).

Applied at the MODALITY level, not the classifier level: dropping modality m
disables the single classifier m AND zero-fills modality m's 3 channels in the
unified input. The unified classifier stays active unless ALL modalities are gone.
Age/sex (tabular) are never dropped. At least one image modality is kept.
"""
from __future__ import annotations

import random

IMG_MODALITIES = ["T1", "T2", "FLAIR", "T1c"]


def apply_modality_dropout(mask: dict, training: bool, p: float = 0.2, rng: random.Random | None = None) -> dict:
    """Return a new modality mask with each present modality dropped w.p. p.

    mask: {modality: bool} — availability from disk. Only True modalities can survive.
    """
    if not training:
        return dict(mask)
    r = rng or random
    out = {m: bool(mask.get(m, False)) and (r.random() > p) for m in IMG_MODALITIES}
    if not any(out.values()):  # keep at least one available modality
        avail = [m for m in IMG_MODALITIES if mask.get(m, False)]
        if avail:
            out[r.choice(avail)] = True
    return out


def classifier_active(modality_mask: dict) -> dict:
    """Map a modality mask to the 6-classifier active dict used by fuse()/losses.

    single classifier m active  <=> modality m present
    unified active               <=> at least one modality present
    tabular always active
    """
    active = {m: bool(modality_mask.get(m, False)) for m in IMG_MODALITIES}
    active["unified"] = any(modality_mask.get(m, False) for m in IMG_MODALITIES)
    active["tabular"] = True
    return active
