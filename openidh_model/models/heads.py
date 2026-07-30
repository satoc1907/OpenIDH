"""Evidence heads (spec §4.2, §4.3).

Both heads emit non-negative evidence via softplus (exp diverges too easily),
clamped away from 0. Convention: output[:, 0] = e1 (evidence for Mut / y=1),
output[:, 1] = e0 (evidence for WT / y=0).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class EvidenceHead(nn.Module):
    """384 -> 2 evidence. Used per single sequence and for the unified trunk."""

    def __init__(self, dim: int = 384, eps: float = 1e-6):
        super().__init__()
        self.fc = nn.Linear(dim, 2)
        self.eps = eps

    def forward(self, x):
        return F.softplus(self.fc(x)).clamp(min=self.eps)


class TabularHead(nn.Module):
    """age, sex -> 2 evidence. Tumor volume is NEVER an input (spec §4.3, §1.2)."""

    def __init__(self, in_dim: int = 2, hidden: int = 64, eps: float = 1e-6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.eps = eps

    def forward(self, x):
        return F.softplus(self.net(x)).clamp(min=self.eps)
