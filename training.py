# training.py
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ExponentialLR,
)

from clip_train_dataset import ClipDataset, make_collate_fn
from motion_clip import MotionClip


# ─────────────────────────────────────────────────────────────────────────────
# Linear warmup wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LinearWarmupScheduler:
    """
    Linearly ramps LR from `warmup_start_lr` to the optimizer's initial LR
    over `warmup_epochs` epochs, then hands off to `after_scheduler`.
    """
    def __init__(self, optimizer, warmup_epochs, warmup_start_lr=1e-8, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.after_scheduler = after_scheduler
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = 0

        # Start at warmup_start_lr
        for group in self.optimizer.param_groups:
            group["lr"] = self.warmup_start_lr

    def step(self):
        self.last_epoch += 1
        if self.last_epoch <= self.warmup_epochs:
            # Linear ramp
            t = self.last_epoch / float(self.warmup_epochs)
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = self.warmup_start_lr + t * (base_lr - self.warmup_start_lr)
        else:
            if self.after_scheduler is not None:
                self.after_scheduler.step()

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def split_by_motion(motion_ids, val_ratio=0.1, seed=0):
    rng = np.random.RandomState(seed)
    motion_ids = np.array(motion_ids)
    perm = rng.permutation(len(motion_ids))
    n_val = int(round(len(motion_ids) * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


@torch.no_grad()
def validate_retrieval(
    motion_clip_model,
    motion_ids,
    motion_paths,
    map_df,
    text_df,
    text_emb,
    device="cuda",
    n_batches=100,
    batch_size=32,
    ks=(1, 2, 3, 5, 10),
):
    """
    Retrieval eval:
      - sample a motion query (one motion id)
      - build candidate text set of size `batch_size` consisting of:
         - one true text for that motion
         - (batch_size-1) random other texts
      - compute query motion embedding and all candidate text embeddings
      - rank, compute hits@k, MRR, etc.

    Returns:
      dict with metrics + "val_score" (mean recall@k) used for early stopping.
    """
    motion_clip_model.eval()
    ks = tuple(sorted(ks))
    max_k = max(ks)

    hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    n = 0

    # motion_id -> list of text_ids
    grouped = map_df.groupby("motion_id")["text_id"].apply(list).to_dict()
    all_text_ids = text_df["text_id"].tolist()
    text_lookup = dict(zip(text_df["text_id"], text_df["text_tensor_id"]))

    for _ in range(n_batches):
        # sample a motion that has at least one text
        motion_id = np.random.choice(motion_ids)
        if motion_id not in grouped:
            continue
        true_text_id = np.random.choice(grouped[motion_id])

        # sample negatives
        negs = []
        while len(negs) < batch_size - 1:
            cand = np.random.choice(all_text_ids)
            if cand != true_text_id:
                negs.append(cand)
        cand_text_ids = [true_text_id] + negs
        np.random.shuffle(cand_text_ids)

        # load motion
        mpath = motion_paths[motion_id]
        motion = np.load(mpath).astype(np.float32)  # (T,D)
        T, D = motion.shape

        # pad to itself (single query) but keep lengths for mask
        motion_pad = torch.zeros((1, T, D), device=device, dtype=torch.float32)
        motion_pad[0] = torch.from_numpy(motion).to(device=device, dtype=torch.float32)
        lengths_t = torch.tensor([T], dtype=torch.long, device=device)

        # motion embedding
        m_emb = F.normalize(motion_clip_model(motion_pad, lengths=lengths_t), dim=-1)  # (1,dim)

        # text embeddings (already computed)
        idxs = [text_lookup[tid] for tid in cand_text_ids]
        t_emb = text_emb[idxs].to(device=device)
        t_emb = F.normalize(t_emb, dim=-1)  # (B,dim)

        sims = (m_emb @ t_emb.t()).squeeze(0)  # (B,)
        rank = torch.argsort(sims, descending=True)

        # position of the true item
        true_pos = cand_text_ids.index(true_text_id)
        true_rank = (rank == true_pos).nonzero(as_tuple=False).item() + 1  # 1-based

        for k in ks:
            if true_rank <= k:
                hits[k] += 1
        mrr_sum += 1.0 / true_rank
        n += 1

    if n == 0:
        return {"val_score": 0.0, "MRR": 0.0, **{f"R@{k}": 0.0 for k in ks}}

    out = {f"R@{k}": hits[k] / n for k in ks}
    out["MRR"] = mrr_sum / n
    out["val_score"] = float(np.mean([out[f"R@{k}"] for k in ks]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLIP loss (A-C hooks live here)
# ─────────────────────────────────────────────────────────────────────────────

def clip_loss(motion_z, text_z, logit_scale, noise_std: float = 0.0):
    # C) tiny embedding noise (train only) to improve generalization
    if noise_std and noise_std > 0.0:
        motion_z = motion_z + noise_std * torch.randn_like(motion_z)
        text_z   = text_z   + noise_std * torch.randn_like(text_z)

    motion_z = F.normalize(motion_z, dim=-1)
    text_z   = F.normalize(text_z,   dim=-1)

    # Keep logits scale sane (also clamped elsewhere on parameter)
    scale    = logit_scale.exp().clamp(1e-3, 20.0)
    logits   = scale * (motion_z @ text_z.t())          # (B, B)
    labels   = torch.arange(logits.size(0), device=logits.device)

    loss_m2t = F.cross_entropy(logits, labels)
    loss_t2m = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_m2t + loss_t2m)


# ─────────────────────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────────────────────

def train_clip_with_split(
    motion_model,
    motion_ids,
    motion_paths,
    map_df,
    text_df,
    text_emb,
    text_model_name,
    save_path,
    val_ratio=0.1,
    seed=0,
    epochs=150,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-3,          # FIX: was 1e-4 – increase to fight overfitting
    n_val_batches=100,          # FIX: was 30 – way too noisy; use ≥100
    val_batch_size=32,          # candidates per retrieval query
    ks=(1, 2, 3, 5, 10),
    patience=50,
    warmup_epochs=5,            # NEW: linear LR warmup
    grad_clip=1.0,              # NEW: gradient clipping (set None to disable)
    logit_scale_max=np.log(20.0),   # B) clamp logit_scale (temperature) upper bound
    freeze_logit_scale_epochs=10,   # A) freeze temperature for first N epochs
    noise_std=1e-3,                 # C) embedding noise (train only)
    time_padding=True,
    augs: dict | None = None,
    scheduler_cfg: dict | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "text_tensor_id" not in text_df.columns:
        text_df = text_df.copy().reset_index(drop=True)
        text_df["text_tensor_id"] = np.arange(len(text_df), dtype=np.int64)

    train_idx, val_idx = split_by_motion(motion_ids, val_ratio=val_ratio, seed=seed)
    train_ids = [motion_ids[i] for i in train_idx]
    val_ids   = [motion_ids[i] for i in val_idx]

    train_ds = ClipDataset(train_ids, motion_paths, map_df, text_df, text_emb, augs=augs)
    collate_fn = make_collate_fn(pad_collate=time_padding)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Build CLIP wrapper
    model = MotionClip(motion_model, text_dim=text_emb.shape[1], text_model_name=text_model_name).to(device)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler (optional)
    after_sched = None
    if scheduler_cfg is not None:
        name = scheduler_cfg.get("name", "").lower()
        if name == "cosine":
            after_sched = CosineAnnealingLR(
                opt,
                T_max=scheduler_cfg.get("T_max", epochs),
                eta_min=scheduler_cfg.get("eta_min", 1e-6),
            )
        elif name == "plateau":
            after_sched = ReduceLROnPlateau(
                opt,
                factor=scheduler_cfg.get("factor", 0.5),
                patience=scheduler_cfg.get("patience", 5),
                min_lr=scheduler_cfg.get("min_lr", 1e-6),
            )
        elif name == "onecycle":
            after_sched = OneCycleLR(
                opt,
                max_lr=scheduler_cfg.get("max_lr", lr),
                total_steps=scheduler_cfg.get("total_steps", epochs * len(train_loader)),
            )

    warmup = LinearWarmupScheduler(opt, warmup_epochs=warmup_epochs, warmup_start_lr=lr * 0.05, after_scheduler=after_sched)

    best = {"val_score": -1.0}
    bad_epochs = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # NaN recovery
    nan_strikes = 0
    nan_strike_limit = 3

    for ep in range(1, epochs + 1):
        # A) Freeze temperature early to avoid overly sharp logits before alignment stabilizes
        if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.nn.Parameter):
            model.logit_scale.requires_grad_(ep > freeze_logit_scale_epochs)

        model.train()
        running = 0.0
        n_steps = 0

        for batch in train_loader:
            # batch can be either (motions, texts) or (motions, lengths, texts)
            if len(batch) == 2:
                motions, texts = batch
                lengths = None
            else:
                motions, lengths, texts = batch

            if time_padding:
                motions = motions.to(device, non_blocking=True)
                if lengths is not None:
                    lengths = lengths.to(device, non_blocking=True)
            else:
                motions = [m.to(device, non_blocking=True) for m in motions]
                if lengths is not None:
                    lengths = lengths.to(device, non_blocking=True)

            texts = texts.to(device, non_blocking=True)

            # ── Clamp logit_scale BEFORE forward so AMP never sees inf/NaN ────
            if hasattr(model, "logit_scale"):
                with torch.no_grad():
                    model.logit_scale.nan_to_num_(nan=np.log(1 / 0.07))
                    model.logit_scale.clamp_(0.0, logit_scale_max)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                motion_z = model(motions, lengths=lengths)
                if not torch.isfinite(motion_z).all():
                    print(f"  [NaN] ep={ep} | motion_z_finite=False | "
                          f"logit_scale={model.logit_scale.item():.4f}")
                    opt.zero_grad(set_to_none=True)
                    nan_strikes += 1
                    continue

                # texts here are already embeddings (B, dim)
                loss = clip_loss(motion_z, texts, model.logit_scale, noise_std=noise_std)

            if not torch.isfinite(loss).all():
                print(f"  [NaN] ep={ep} | loss not finite | "
                      f"logit_scale={model.logit_scale.item():.4f}")
                opt.zero_grad(set_to_none=True)
                nan_strikes += 1
                continue

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            # Clamp logit_scale after update as well
            if hasattr(model, "logit_scale"):
                with torch.no_grad():
                    model.logit_scale.nan_to_num_(nan=np.log(1 / 0.07))
                    model.logit_scale.clamp_(0.0, logit_scale_max)

            running += loss.item()
            n_steps += 1

            if nan_strikes >= nan_strike_limit:
                break

        # Warmup / scheduler step
        warmup.step()
        cur_lr = warmup.get_last_lr()[0]

        train_loss = running / max(1, n_steps)

        # Validation
        metrics = validate_retrieval(
            motion_clip_model=model,
            motion_ids=val_ids,
            motion_paths=motion_paths,
            map_df=map_df,
            text_df=text_df,
            text_emb=text_emb,
            device=device,
            n_batches=n_val_batches,
            batch_size=val_batch_size,
            ks=ks,
        )

        val_score = metrics["val_score"]
        mrr = metrics["MRR"]
        rks = " ".join([f"R@{k}={metrics[f'R@{k}']:.3f}" for k in ks])

        print(
            f"epoch {ep:03d}/{epochs} | lr={cur_lr:.2e} | "
            f"train_loss={train_loss:.4f} | val_score={val_score:.4f} | "
            f"MRR={mrr:.4f} | {rks}"
        )

        # Early stopping
        if val_score > best["val_score"]:
            best = {"val_score": val_score, "epoch": ep, "metrics": metrics}
            bad_epochs = 0

            # Only save if logit_scale is healthy — NaN logit_scale would corrupt checkpoint
            ls_healthy = True
            if hasattr(model, "logit_scale"):
                ls_healthy = torch.isfinite(model.logit_scale).item()
            if ls_healthy:
                torch.save(model.state_dict(), save_path)
                print(f"Modèle sauvegardé (Nouveau meilleur score : {val_score:.4f})")
                print(f"Found New Best Model! (MRR={mrr:.4f})")
        else:
            bad_epochs += 1
            print(f"EarlyStopping counter: {bad_epochs} out of {patience}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {ep}.")
            break

        # NaN strike handling: reload best model, halve LR
        if nan_strikes >= nan_strike_limit:
            print(f"⚠️  NaN in epoch {ep} (strike {nan_strikes}/{nan_strike_limit}). "
                  f"Reloading checkpoint from disk and halving LR.")
            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path, map_location=device))
            for group in opt.param_groups:
                group["lr"] *= 0.5
            nan_strikes = 0

    # Load best weights for return
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model.motion_model, best
