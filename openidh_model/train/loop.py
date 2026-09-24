"""Training loop with the mandated op order and inner-val early stopping.

Op order per step (spec §4.1): raw evidence -> volume scale (image only) ->
modality dropout -> fuse -> loss. Early stopping is on inner-val NLL (spec §9).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import MODALITIES, GliomaDataset
from ..data.metadata import lookup_age_sex
from ..eval.metrics import compute_metrics
from ..losses.evidential import total_loss
from ..models.fusion import fuse
from ..models.openidh import OpenIDH


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_kept(avail: dict, p: float, training: bool, device, gen) -> dict:
    """Per-sample modality keep-mask; drops w.p. p, keeps >=1 available (spec §7)."""
    kept = {}
    B = next(iter(avail.values())).shape[0]
    if not training:
        return {m: avail[m].to(device) for m in MODALITIES}
    for m in MODALITIES:
        roll = torch.rand(B, generator=gen, device=device) > p
        kept[m] = avail[m].to(device) & roll
    stack = torch.stack([kept[m] for m in MODALITIES], 0)  # (4, B)
    none_left = ~stack.any(0)                               # (B,)
    if none_left.any():
        avail_stack = torch.stack([avail[m].to(device) for m in MODALITIES], 0)
        for b in torch.nonzero(none_left, as_tuple=False).flatten():
            cand = torch.nonzero(avail_stack[:, b], as_tuple=False).flatten()
            if len(cand):
                kept[MODALITIES[int(cand[0])]][b] = True
    return kept


def _forward(model, batch, cfg, training, device, gen, drop=None):
    """One step in the mandated op order. `drop` names modalities to treat as
    NOT ACQUIRED (eval-time ablation, spec §7): their channels are zeroed — so
    the unified trunk sees the absence too — and their heads leave the fusion,
    exactly as a genuinely missing sequence would (spec §3.5). It is applied
    after the dropout mask, so dropping is idempotent with it."""
    images = {m: batch["images"][m].to(device) for m in MODALITIES}
    avail = {m: batch["mask"][m] for m in MODALITIES}
    volume = batch["volume"].to(device)
    tabular = batch["tabular"].to(device)
    y = batch["label"].to(device)

    kept = _sample_kept(avail, cfg["train"]["modality_dropout_p"], training, device, gen)
    for m in drop or ():
        kept[m] = torch.zeros_like(kept[m])
    for m in MODALITIES:  # zero dropped modality channels (also feeds the unified trunk)
        images[m] = images[m] * kept[m].view(-1, 1, 1, 1).to(images[m].dtype)

    ev = model.raw_evidence(images, tabular)
    active = {m: kept[m] for m in MODALITIES}
    active["unified"] = torch.stack([kept[m] for m in MODALITIES], 0).any(0)
    active["tabular"] = torch.ones_like(y, dtype=torch.bool)

    alpha, beta, scaled = fuse(ev, active, volume,
                               cfg["fusion"]["v_ref"], cfg["fusion"]["gamma"])
    return ev, active, alpha, beta, scaled, y


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    gen = torch.Generator(device=device)
    A, Bb, Y = [], [], []
    ev_sums = {n: [] for n in ["T1", "T2", "FLAIR", "T1c", "unified", "tabular"]}
    fd = fr = n = 0.0
    for batch in loader:
        _, active, alpha, beta, scaled, y = _forward(model, batch, cfg, False, device, gen)
        A.append(alpha.cpu()); Bb.append(beta.cpu()); Y.append(y.cpu())
        for name, e in scaled.items():
            ev_sums[name].append((e[:, 0] + e[:, 1]).cpu())
        _, comp = total_loss(alpha, beta, scaled, active, y,
                             cfg["loss"]["lambda_reg_max"], cfg["loss"]["lambda_aux"])
        bs = y.shape[0]
        fd += float(comp["fused_data"]) * bs
        fr += float(comp["fused_reg"]) * bs
        n += bs
    alpha = torch.cat(A).numpy(); beta = torch.cat(Bb).numpy(); y = torch.cat(Y).numpy()
    metrics = compute_metrics(alpha, beta, y)
    records = {
        "S": alpha + beta, "p": alpha / (alpha + beta), "y": y,
        "evidence": {k: torch.cat(v).numpy() for k, v in ev_sums.items()},
        "fused_data": fd / n, "fused_reg": fr / n,
    }
    return metrics, records


def _balanced_subset(rows, n):
    mut = [r for r in rows if str(r["idh"]).strip() == "Mut"]
    wt = [r for r in rows if str(r["idh"]).strip() == "WT"]
    k = n // 2
    return (mut[:k] + wt[: n - min(k, len(mut))])[:n]


def make_loader(ds, cfg, shuffle: bool):
    """DataLoader with cfg-driven workers.

    Each __getitem__ gunzips 5 NIfTI volumes, so loading — not the GPU — is the
    bottleneck on the cluster; `train.num_workers` parallelises it. __getitem__ has
    no RNG and shuffling stays in the main-process sampler, so the worker count does
    not change the data order. 0 keeps the single-process path (local).

    Measured on the smoke fold (nw=0 vs nw=4, seed 0): identical epoch-by-epoch, and
    bitwise identical once OMP_NUM_THREADS is pinned. Unpinned, metrics differ by
    ~1e-5 relative — worker processes shift the main process's OpenMP thread count,
    which reorders float reductions. Nothing to do with the data pipeline.
    """
    nw = int(cfg["train"].get("num_workers", 0) or 0)
    # NOT persistent_workers: a persistent iterator draws the worker base seed only
    # once instead of once per epoch, which shifts the global RNG stream and changes
    # the shuffle order from epoch 1 on (measured: smoke diverges at epoch 1).
    pf = int(cfg["train"].get("prefetch_factor", 1) or 1)
    extra = {"prefetch_factor": pf} if nw > 0 else {}
    if nw > 0:
        _warn_if_shm_tight(cfg, nw, pf)
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle,
                      num_workers=nw, **extra)


def _shm_budget(cfg, nw: int, pf: int) -> tuple[float, float]:
    """(bytes of image tensors in flight, bytes free in /dev/shm)."""
    import shutil

    per_subject = len(MODALITIES) * 3 * cfg["data"]["image_size"] ** 2 * 4   # float32
    in_flight = nw * pf * cfg["train"]["batch_size"] * per_subject
    try:
        free = float(shutil.disk_usage("/dev/shm").free)
    except OSError:
        free = float("inf")
    return float(in_flight), free


def _warn_if_shm_tight(cfg, nw: int, pf: int) -> None:
    """Worker batches are handed over through /dev/shm, which the cluster container
    caps at 1GB (--shm-size=1g). Exceed it and the run dies partway through with
    "unable to allocate shared memory" — non-deterministically, so sibling seeds can
    survive while one does not. BOTH of torch's sharing strategies allocate there
    (file_system was tried and failed the same way), so the only real lever is how
    much is in flight: num_workers x prefetch_factor x batch_size x 2.4MB/subject.
    Surfaced at startup rather than twenty minutes in.
    """
    import warnings

    need, free = _shm_budget(cfg, nw, pf)
    if need > 0.8 * free:
        warnings.warn(
            f"dataloader may exhaust /dev/shm: {need/1e6:.0f}MB in flight "
            f"({nw} workers x prefetch {pf} x batch {cfg['train']['batch_size']}) "
            f"against {free/1e6:.0f}MB free — lower train.prefetch_factor or "
            f"train.num_workers", RuntimeWarning, stacklevel=2)


def train_fold(paths, cfg, split_file, fold_id=None, seed=0, log=print):
    set_seed(seed)
    device = cfg["train"]["device"]
    subset_n = cfg["data"].get("subset_n")

    train_ds = GliomaDataset(paths.splits_dir / split_file, paths, role="train", fold_id=fold_id)
    val_ds = GliomaDataset(paths.splits_dir / split_file, paths, role="val", fold_id=fold_id)
    if subset_n:
        train_ds.rows = _balanced_subset(train_ds.rows, subset_n)
        val_ds.rows = _balanced_subset(val_ds.rows, max(4, subset_n // 2))

    ages = np.array([lookup_age_sex(paths.metadata_dir, r["site"], r["subject_id"])[0]
                     for r in train_ds.rows], dtype=np.float64)
    age_mean = float(np.nanmean(ages)); age_std = float(np.nanstd(ages)) or 1.0
    for ds in (train_ds, val_ds):
        ds.age_mean, ds.age_std = age_mean, age_std
    log(f"train={len(train_ds)} val={len(val_ds)} age_mean={age_mean:.1f} age_std={age_std:.1f}")

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    model = OpenIDH(backbone=cfg["model"]["backbone"], weights_dir=paths.weights_dir).to(device)
    # lr_trunk/lr_head default to lr, which reproduces the single-rate optimizer
    lr = cfg["train"]["lr"]
    lr_trunk = float(cfg["train"].get("lr_trunk") or lr)
    lr_head = float(cfg["train"].get("lr_head") or lr)
    opt = torch.optim.AdamW(model.param_groups(lr_trunk, lr_head), lr=lr,
                            weight_decay=cfg["train"]["weight_decay"])
    if lr_trunk != lr_head:
        log(f"lr split: trunk={lr_trunk:g} head={lr_head:g}")
    gen = torch.Generator(device=device); gen.manual_seed(seed)

    lam_max = cfg["loss"]["lambda_reg_max"]; t_anneal = cfg["loss"]["t_anneal"]
    lam_aux = cfg["loss"]["lambda_aux"]
    best = {"nll": float("inf"), "epoch": -1}; history = []; bad = 0

    for epoch in range(cfg["train"]["epochs"]):
        t0 = time.perf_counter()
        model.train()
        lam_reg_t = lam_max * min(1.0, epoch / max(1, t_anneal))
        tr_loss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            _, active, alpha, beta, scaled, y = _forward(model, batch, cfg, True, device, gen)
            loss, _ = total_loss(alpha, beta, scaled, active, y, lam_reg_t, lam_aux)
            loss.backward(); opt.step()
            tr_loss += float(loss.detach()) * y.shape[0]
        tr_loss /= max(1, len(train_ds))

        metrics, records = evaluate(model, val_loader, cfg, device)
        secs = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": tr_loss, "lam_reg_t": lam_reg_t,
                        "secs": secs, **metrics})
        log(f"epoch {epoch}: train_loss={tr_loss:.4f} val_nll={metrics['nll']:.4f} "
            f"val_auc={metrics['auc']:.3f} median_S={metrics['median_S']:.1f} "
            f"({secs:.1f}s)")

        if metrics["nll"] < best["nll"]:
            best = {"nll": metrics["nll"], "epoch": epoch, "state": {k: v.cpu().clone()
                    for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["train"]["early_stop_patience"]:
                log(f"early stop at epoch {epoch} (best epoch {best['epoch']})")
                break

    return {"model": model, "history": history, "best": best,
            "train_ds": train_ds, "val_ds": val_ds,
            "age_mean": age_mean, "age_std": age_std}
