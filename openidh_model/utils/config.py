"""Config loading: merge base.yaml + model.yaml + a train yaml (spec §11).

`train.smoke.yaml` holds only diffs from `train.yaml`, so it is overlaid on top.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(configs_dir, train_yaml: str = "train.yaml") -> dict:
    configs_dir = Path(configs_dir)
    cfg: dict = {}
    for f in ("base.yaml", "model.yaml", "train.yaml"):
        _deep_update(cfg, yaml.safe_load((configs_dir / f).read_text()))
    if train_yaml != "train.yaml":  # smoke (or any variant) overlays train.yaml
        _deep_update(cfg, yaml.safe_load((configs_dir / train_yaml).read_text()))
    return cfg
