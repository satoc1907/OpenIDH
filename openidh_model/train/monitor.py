"""Training monitors (spec §13, §6.2.1).

Summarizes an eval pass so we can see whether the design is actually working —
not just whether AUC is high. The key signal is `S_correct` vs `S_wrong`:
if the model is not less confident when wrong, the uncertainty is worthless.
"""
from __future__ import annotations

import numpy as np

CLASSIFIERS = ["T1", "T2", "FLAIR", "T1c", "unified", "tabular"]


def summarize(records: dict) -> dict:
    """records: arrays S, p, y (N,) and evidence {name: (N,)} of scaled e1+e0."""
    S = np.asarray(records["S"])
    p = np.asarray(records["p"])
    y = np.asarray(records["y"])
    pred = (p > 0.5).astype(int)
    correct = pred == y.astype(int)

    ev = records.get("evidence", {})
    balance = {n: float(np.mean(ev[n])) for n in CLASSIFIERS if n in ev}

    out = {
        "median_S": float(np.median(S)),
        "S_correct_median": float(np.median(S[correct])) if correct.any() else float("nan"),
        "S_wrong_median": float(np.median(S[~correct])) if (~correct).any() else float("nan"),
        "evidence_balance": balance,
    }
    if "fused_data" in records and "fused_reg" in records:
        fd, fr = records["fused_data"], records["fused_reg"]
        out["fused_data"] = float(fd)
        out["fused_reg"] = float(fr)
        out["reg_over_data"] = float(fr / fd) if fd > 0 else float("inf")

    # design signal: are we less confident when wrong?
    out["uncertainty_ok"] = bool(
        np.isnan(out["S_wrong_median"])
        or np.isnan(out["S_correct_median"])
        or out["S_wrong_median"] <= out["S_correct_median"]
    )
    return out


def format_summary(s: dict) -> str:
    lines = [
        f"  median S              : {s['median_S']:.1f}   (target 10–120)",
        f"  S (correct) median    : {s['S_correct_median']:.1f}",
        f"  S (wrong)   median    : {s['S_wrong_median']:.1f}   "
        f"{'OK <= correct' if s['uncertainty_ok'] else 'WARN: wrong not less confident'}",
    ]
    if "reg_over_data" in s:
        lines.append(f"  fused reg/data ratio  : {s['reg_over_data']:.2f}  "
                     f"(data={s['fused_data']:.3f}, reg={s['fused_reg']:.3f})")
    bal = ", ".join(f"{k}={v:.2f}" for k, v in s["evidence_balance"].items())
    lines.append(f"  evidence balance      : {bal}")
    return "\n".join(lines)
