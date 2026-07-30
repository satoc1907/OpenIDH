"""Download the pretrained ViT weights to weights/ ONCE, locally (spec §1.4).

The HPC run is offline, so pretrained weights must be staged as a local
safetensors file. Run this once on a networked machine:

    uv run python scripts/download_weights.py

The raw state_dict (WITH the classification head) is saved; num_classes / in_chans
adaptation happens at build time via `pretrained_cfg_overlay` (see encoder.build_trunk).
"""
from __future__ import annotations

from pathlib import Path

import timm
from safetensors.torch import save_file

MODEL = "vit_small_patch16_224.augreg_in21k"


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "weights"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{MODEL}.safetensors"

    m = timm.create_model(MODEL, pretrained=True)  # keep default head
    state = {k: v.contiguous() for k, v in m.state_dict().items()}
    save_file(state, str(dst))

    print(f"saved: {dst}")
    print(f"param count (with head): {sum(p.numel() for p in m.parameters()):,}")


if __name__ == "__main__":
    main()
