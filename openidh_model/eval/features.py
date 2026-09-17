"""CLS-feature dump for the shift decomposition (experiment A).

The model is late-fusion: there is no single post-fusion representation, only
five 384-d CLS features (one per image encoder) each feeding a Linear(384->2)
evidence head. To ask *where in feature space* a site shift lands, we need those
five features for every subject, plus the head weights that turn them into
evidence. This module captures them from the SAME forward pass the model uses
for evaluation (forward hooks on the two trunks), so nothing is re-implemented.

  features_{split}.npz   feat_{enc} (N,384) float32, active_{enc} (N,) bool,
                         evidence_{enc} (N,2) raw pre-scaling evidence, plus
                         subject_id, site, y, age, sex, grade, volume
  head_weights.npz       W_{enc} (2,384), b_{enc} (2,)

Row convention of every head (models/heads.py): row 0 -> e1 (Mut, y=1),
row 1 -> e0 (WT, y=0). `verify_heads` recomputes the evidence from the dumped
features and weights and checks it against the model's own output, so a swapped
row would be caught before any decomposition runs on it.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.metadata import lookup_age_sex, lookup_grade
from ..models.openidh import IMG_MODALITIES

ENCODERS = IMG_MODALITIES + ["unified"]   # T1, T2, FLAIR, T1c, unified
E1_ROW, E0_ROW = 0, 1                     # head output rows (models/heads.py)


@contextmanager
def capture_cls(model):
    """Record the trunk outputs of the next forward pass.

    trunk_single is called once per modality in IMG_MODALITIES order inside
    raw_evidence, so its hook fills a list that we zip with that order.
    """
    store: dict = {"single": [], "unified": None}
    h1 = model.trunk_single.register_forward_hook(
        lambda m, i, o: store["single"].append(o.detach()))
    h2 = model.trunk_unified.register_forward_hook(
        lambda m, i, o: store.__setitem__("unified", o.detach()))
    try:
        yield store
    finally:
        h1.remove(); h2.remove()


def cls_from_store(store: dict) -> dict:
    if len(store["single"]) != len(IMG_MODALITIES) or store["unified"] is None:
        raise RuntimeError("capture_cls saw an unexpected number of trunk calls")
    out = {m: store["single"][i] for i, m in enumerate(IMG_MODALITIES)}
    out["unified"] = store["unified"]
    return out


def head_of(model, enc: str):
    return model.head_unified if enc == "unified" else model.heads_single[enc]


def head_weights(model) -> dict:
    """{enc: (W (2,384), b (2,))} as float32 numpy, in the head's own row order."""
    out = {}
    for enc in ENCODERS:
        fc = head_of(model, enc).fc
        out[enc] = (fc.weight.detach().cpu().numpy().astype(np.float32),
                    fc.bias.detach().cpu().numpy().astype(np.float32))
    return out


def evidence_from_features(feat: np.ndarray, W: np.ndarray, b: np.ndarray,
                           eps: float = 1e-6) -> np.ndarray:
    """Hand-computed head: softplus(W f + b).clamp(eps) -> (N, 2) [e1, e0]."""
    z = torch.from_numpy(np.asarray(feat, dtype=np.float32)) @ torch.from_numpy(W).T \
        + torch.from_numpy(b)
    return F.softplus(z).clamp(min=eps).numpy()


@torch.no_grad()
def collect_features(model, loader, cfg, device, paths) -> dict:
    """Run the evaluation forward pass (eval mode, no dropout, no augmentation)
    and return CLS features + raw evidence per encoder, with subject covariates.

    Volume scaling is an evidence-stage operation and never touches the features,
    so it is neither applied nor needed here. The tabular head has no CLS feature
    and is out of scope.
    """
    from ..train.loop import _forward  # local import: loop imports eval.metrics

    model.eval()
    gen = torch.Generator(device=device)
    cols: dict = {f"feat_{e}": [] for e in ENCODERS}
    cols.update({f"active_{e}": [] for e in ENCODERS})
    cols.update({f"evidence_{e}": [] for e in ENCODERS})
    meta = {k: [] for k in ("subject_id", "site", "y", "age", "sex", "grade", "volume")}

    for batch in loader:
        with capture_cls(model) as store:
            ev, active, _, _, _, y = _forward(model, batch, cfg, False, device, gen)
        feats = cls_from_store(store)
        for e in ENCODERS:
            cols[f"feat_{e}"].append(feats[e].float().cpu().numpy())
            cols[f"active_{e}"].append(active[e].cpu().numpy().astype(bool))
            cols[f"evidence_{e}"].append(ev[e].float().cpu().numpy())
        for i in range(len(y)):
            sid, site = batch["subject_id"][i], batch["site"][i]
            age, sex = lookup_age_sex(paths.metadata_dir, site, sid)   # raw years, not standardised
            meta["subject_id"].append(sid); meta["site"].append(site)
            meta["y"].append(int(y[i].item()))
            meta["age"].append(float(age)); meta["sex"].append(float(sex))
            meta["grade"].append(lookup_grade(paths.metadata_dir, site, sid))
            meta["volume"].append(float(batch["volume"][i]))

    out = {k: np.concatenate(v, 0) for k, v in cols.items()}
    out["subject_id"] = np.array(meta["subject_id"], dtype=str)
    out["site"] = np.array(meta["site"], dtype=str)
    out["y"] = np.array(meta["y"], dtype=np.int8)
    for k in ("age", "sex", "grade", "volume"):
        out[k] = np.array(meta[k], dtype=np.float32)
    return out


def verify_heads(feats: dict, weights: dict, atol: float = 1e-4) -> dict:
    """Recompute every head from (features, W, b) and compare with the model's
    evidence. Returns the max abs error per encoder; raises if any exceeds atol.

    This is the row-order check the analysis depends on: if W's rows were
    swapped, e1 and e0 would trade places and every d_p sign would flip.
    """
    err = {}
    for e in ENCODERS:
        W, b = weights[e]
        hand = evidence_from_features(feats[f"feat_{e}"], W, b)
        err[e] = float(np.abs(hand - feats[f"evidence_{e}"]).max())
    bad = {e: v for e, v in err.items() if v > atol}
    if bad:
        raise AssertionError(f"head recomputation mismatch (row order?): {bad}")
    return err


def write_features(path: str | Path, feats: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **feats)
    return path


def write_head_weights(path: str | Path, weights: dict) -> Path:
    path = Path(path)
    arrays = {}
    for e, (W, b) in weights.items():
        arrays[f"W_{e}"] = W; arrays[f"b_{e}"] = b
    arrays["row_order"] = np.array(["e1(Mut,y=1)", "e0(WT,y=0)"], dtype=str)
    np.savez(path, **arrays)
    return path


def load_features(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_head_weights(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as z:
        assert list(z["row_order"]) == ["e1(Mut,y=1)", "e0(WT,y=0)"], z["row_order"]
        return {e: (z[f"W_{e}"], z[f"b_{e}"]) for e in ENCODERS}
