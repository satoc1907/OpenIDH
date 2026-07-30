"""ViT trunk construction (spec §1.3, §4.1).

Weights are read from a local safetensors file so the model builds with no
network access (spec §1.4). timm still applies `adapt_input_conv` for `in_chans=12`
and drops the classification head (`num_classes=0`) as usual.
"""
from __future__ import annotations

from pathlib import Path

import timm


def build_trunk(backbone: str, in_chans: int, weights_dir, pretrained: bool = True):
    """Build a headless ViT trunk from local pretrained weights.

    in_chans=3  -> shared trunk for the 4 single sequences.
    in_chans=12 -> independent trunk for the unified (4seq x 3slice) input.
    """
    ckpt = Path(weights_dir) / f"{backbone}.safetensors"
    # custom_load=False forces timm's generic state_dict loader (this backbone's
    # default is an npz "custom_load" path that can't read our safetensors). The
    # generic path still applies adapt_input_conv (in_chans=12) and drops the head.
    overlay = dict(file=str(ckpt), custom_load=False) if ckpt.exists() else None
    if pretrained and overlay is None:
        raise FileNotFoundError(
            f"local weights not found: {ckpt}. Run scripts/download_weights.py first "
            "(the HPC run is offline; weights must be staged locally — spec §1.4)."
        )
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=0,
        in_chans=in_chans,
        pretrained_cfg_overlay=overlay,
    )
