"""Per-subject prediction dump (spec §14.2 / §6.1 follow-up analyses).

`result.json` only keeps aggregates, so every follow-up question — reliability
diagrams, prior correction, in-site vs out-site evidence — used to need another
inference pass. This writes one row per subject instead, with the fused Beta and
the per-head evidence that produced it, so those analyses become pure CSV work.

Columns (see PRED_COLUMNS):
  role         inner-val or test — the same model scored on both
  alpha, beta  fused Beta; p_hat = alpha/(alpha+beta), S = alpha+beta
  e1_*, e0_*   post-scaling, post-masking evidence per head (fusion.fuse)
  active_mask  one 0/1 per head in CLASSIFIERS order, e.g. "111011"
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from ..train.monitor import CLASSIFIERS

PRED_COLUMNS = (
    ["run", "role", "subject_id", "site", "y_true", "alpha", "beta", "p_hat", "S"]
    + [f"{k}_{h}" for h in CLASSIFIERS for k in ("e1", "e0")]
    + ["volume", "active_mask"]
)


@torch.no_grad()
def collect_predictions(model, loader, cfg, device, role: str, run: str = "",
                        drop=None) -> list[dict]:
    """One row per subject. Inference only — modality dropout is off (training=False).

    `drop` (a sequence of modality names) evaluates the model as if those
    sequences had never been acquired: zeroed channels into the unified trunk
    and their heads out of the fusion (see train.loop._forward). The rows are
    written in the same schema, so `active_mask` records what was used.
    """
    from ..train.loop import _forward  # local import: loop imports eval.metrics

    model.eval()
    gen = torch.Generator(device=device)
    rows: list[dict] = []
    for batch in loader:
        _, active, alpha, beta, scaled, y = _forward(model, batch, cfg, False, device, gen,
                                                     drop=drop)
        S = (alpha + beta).cpu()
        p = (alpha / (alpha + beta)).cpu()
        a, b = alpha.cpu(), beta.cpu()
        vol = batch["volume"].cpu()
        act = {h: active[h].cpu() for h in CLASSIFIERS}
        ev = {h: scaled[h].cpu() for h in CLASSIFIERS}
        for i in range(len(y)):
            r = {
                "run": run, "role": role,
                "subject_id": batch["subject_id"][i], "site": batch["site"][i],
                "y_true": int(y[i].item()),
                # 6dp keeps every digit that matters and roughly halves the file
                "alpha": round(float(a[i]), 6), "beta": round(float(b[i]), 6),
                "p_hat": round(float(p[i]), 6), "S": round(float(S[i]), 6),
                "volume": int(vol[i]),
                "active_mask": "".join("1" if bool(act[h][i]) else "0" for h in CLASSIFIERS),
            }
            for h in CLASSIFIERS:
                r[f"e1_{h}"] = round(float(ev[h][i, 0]), 6)
                r[f"e0_{h}"] = round(float(ev[h][i, 1]), 6)
            rows.append(r)
    return rows


def write_predictions(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRED_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path
