"""Evaluation metrics (spec §6.1).

Works on fused Beta(alpha, beta). p_hat = alpha/(alpha+beta), S = alpha+beta.
Early stopping uses NLL (spec §9); AUC is threshold-free (spec §5.3).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def compute_metrics(alpha, beta, y) -> dict:
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    S = a + b
    p = a / S
    eps = 1e-7
    pc = np.clip(p, eps, 1 - eps)
    nll = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    out = {
        "nll": nll,
        "brier": float(np.mean((p - y) ** 2)),
        "ece": _ece(p, y),
        "mean_S": float(S.mean()),
        "median_S": float(np.median(S)),
    }
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:  # single-class batch (can happen in a tiny smoke run)
        out["auc"] = float("nan")
        out["auprc"] = float("nan")
    return out
