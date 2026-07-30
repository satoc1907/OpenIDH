"""Evidential loss (spec §3, §6).

Data term = Bayes risk of MSE (bias + variance). Regularizer = KL to the uniform
Beta(1,1) with the correct-class evidence collapsed, which for binary reduces to
the digamma-free closed form  log(w) - (w-1)/w,  w = e_wrong + 1  (spec §3.3.6).
float32 is sufficient; no lgamma/digamma.
"""
from __future__ import annotations

import torch


def loss_data(alpha: torch.Tensor, beta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bayes risk of MSE: (y - p)^2 + p(1-p)/(S+1)."""
    S = alpha + beta
    p = alpha / S
    return (y - p) ** 2 + p * (1.0 - p) / (S + 1.0)


def loss_reg(e1: torch.Tensor, e0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """KL(Beta(tilde) || Beta(1,1)) — penalizes only wrong-direction evidence.

    Binary closed form: log(w) - (w-1)/w, with w = e_wrong + 1.
    y=1 -> wrong direction is e0; y=0 -> wrong direction is e1.
    """
    e_wrong = torch.where(y > 0.5, e0, e1)
    w = e_wrong + 1.0
    return torch.log(w) - (w - 1.0) / w


def total_loss(alpha_f, beta_f, scaled, active, y, lam_reg_t, lam_aux):
    """Fused loss + lam_aux * mean-over-active auxiliary loss (spec §6.3).

    The fused evidence is NOT normalized by |A| (done in fusion), but the
    auxiliary loss IS averaged over active classifiers (spec §3.1 asymmetry).
    Returns (scalar_loss, components_dict) with fused_data / fused_reg logged
    separately (the reg term can dominate the fused side — spec §3.1).
    """
    l_fused_data = loss_data(alpha_f, beta_f, y)
    l_fused_reg = loss_reg(alpha_f - 1.0, beta_f - 1.0, y)
    l_fused = l_fused_data + lam_reg_t * l_fused_reg

    l_aux = torch.zeros_like(l_fused)
    n_active = torch.zeros_like(l_fused)
    for name, e in scaled.items():
        a, b = e[:, 0] + 1.0, e[:, 1] + 1.0
        li = loss_data(a, b, y) + lam_reg_t * loss_reg(e[:, 0], e[:, 1], y)
        m = active[name].to(li.dtype)
        l_aux = l_aux + li * m
        n_active = n_active + m
    l_aux = l_aux / n_active.clamp(min=1.0)

    loss = (l_fused + lam_aux * l_aux).mean()
    return loss, {
        "fused_data": l_fused_data.mean().detach(),
        "fused_reg": l_fused_reg.mean().detach(),
        "aux": l_aux.mean().detach(),
    }
