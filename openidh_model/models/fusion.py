"""Volume-based evidence scaling and evidence fusion (spec §2, §5).

Two invariants that are easy to get wrong (spec §0):
  * fuse SUMS evidence then +1 — never sum alpha (all-missing must give Beta(1,1)).
  * the fused evidence is NOT divided by |A| — dividing kills the "less info ->
    wider tail" property (the auxiliary loss IS divided; that lives in losses/).
"""
from __future__ import annotations

import torch

V_REF = 10_000  # voxel = 10 cm^3 (UTSW p5 = 9,497; protect only the smallest ~5%)
GAMMA = 0.5

# Classifiers whose evidence is volume-scaled (the 5 image classifiers).
# The tabular classifier is never scaled.
TABULAR = "tabular"


def volume_scale(volume: torch.Tensor, v_ref: float = V_REF, gamma: float = GAMMA) -> torch.Tensor:
    """s(V) = min(1, (V / V_ref)^gamma). Shape-preserving, in [0, 1]."""
    v = volume.to(torch.float32)
    return torch.clamp((v / v_ref) ** gamma, max=1.0)


def fuse(
    evidences: dict,
    active: dict,
    volume: torch.Tensor,
    v_ref: float = V_REF,
    gamma: float = GAMMA,
):
    """Scale image evidence by s(V), zero out inactive classifiers, sum, +1.

    evidences: {name: (B, 2)} raw evidence ([e1, e0]).
    active:    {name: (B,)}   bool — False for a missing/dropped classifier.
    volume:    (B,) whole-tumor voxel count.
    returns (alpha (B,), beta (B,), scaled {name: (B,2)}), where `scaled` is the
    post-scaling, post-masking evidence used by the auxiliary loss.
    """
    s = volume_scale(volume, v_ref, gamma).unsqueeze(-1)  # (B, 1)

    scaled = {}
    for name, e in evidences.items():
        e = e if name == TABULAR else e * s          # scale image classifiers only
        m = active[name].unsqueeze(-1).to(e.dtype)   # (B, 1) 0/1
        scaled[name] = e * m                         # inactive -> 0

    e_sum = sum(scaled.values())                     # (B, 2); NOT normalized by |A|
    alpha = e_sum[:, 0] + 1.0
    beta = e_sum[:, 1] + 1.0
    return alpha, beta, scaled
