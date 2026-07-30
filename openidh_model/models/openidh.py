"""The OpenIDH module — 6 classifiers producing raw evidence (spec §1.1, §4.1).

Trunk sharing (spec §1.3): the 4 single sequences share ONE 3-channel trunk
(seen 4x per step -> data efficiency); the 12-channel unified input has its own
trunk. Each single sequence has its own evidence head (case B), plus one head for
the unified trunk and a tabular MLP.

`raw_evidence` returns evidence BEFORE volume scaling / fusion, so tumor volume
never touches any classifier input (spec §1.2 pitfall #3). Scaling + fusion live
in fusion.py (Stage A).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import build_trunk
from .heads import EvidenceHead, TabularHead

# Fixed channel order for the unified 12ch input (spec §3.3 — order must not change).
IMG_MODALITIES = ["T1", "T2", "FLAIR", "T1c"]


class OpenIDH(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_small_patch16_224.augreg_in21k",
        weights_dir="weights",
        embed_dim: int = 384,
        pretrained: bool = True,
    ):
        super().__init__()
        # shared trunk for the 4 single sequences (3ch) ...
        self.trunk_single = build_trunk(backbone, 3, weights_dir, pretrained)
        # ... independent trunk for the unified 12ch input.
        self.trunk_unified = build_trunk(backbone, 12, weights_dir, pretrained)

        # 5 evidence heads (one per single sequence + one for the unified trunk).
        self.heads_single = nn.ModuleDict(
            {m: EvidenceHead(embed_dim) for m in IMG_MODALITIES}
        )
        self.head_unified = EvidenceHead(embed_dim)

        # tabular classifier: age, sex only.
        self.head_tabular = TabularHead(in_dim=2)

    def raw_evidence(self, images: dict, tabular: torch.Tensor) -> dict:
        """Raw (pre-scaling, pre-fusion) evidence for each of the 6 classifiers.

        images:  {modality: (B, 3, H, W)} for T1/T2/FLAIR/T1c.
        tabular: (B, 2) = [age_standardized, sex].
        returns: {name: (B, 2)} for names T1,T2,FLAIR,T1c,unified,tabular.
        """
        ev = {}
        for m in IMG_MODALITIES:
            ev[m] = self.heads_single[m](self.trunk_single(images[m]))
        x12 = torch.cat([images[m] for m in IMG_MODALITIES], dim=1)  # (B, 12, H, W)
        ev["unified"] = self.head_unified(self.trunk_unified(x12))
        ev["tabular"] = self.head_tabular(tabular)
        return ev

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> dict:
        def n(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "trunk_single(3ch)": n(self.trunk_single),
            "trunk_unified(12ch)": n(self.trunk_unified),
            "evidence_heads(x5)": sum(n(h) for h in self.heads_single.values())
            + n(self.head_unified),
            "tabular_mlp": n(self.head_tabular),
            "total": self.num_parameters(),
        }
