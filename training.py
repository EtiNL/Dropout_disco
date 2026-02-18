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

    Usage:
        scheduler = LinearWarmupScheduler(opt, warmup_epochs=5, after_scheduler=cosine_sched)
        # each epoch: scheduler.step()   (pass val_score for plateau)
    """
    def __init__(self, optimizer, warmup_epochs: int, after_scheduler,
                 after_scheduler_kind: str = "epoch",
                 warmup_start_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.after_scheduler_kind = after_scheduler_kind  # "epoch" | "plateau" | "batch"
        self.warmup_start_lr = warmup_start_lr
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._epoch = 0
        # set initial LR to warmup_start
        for pg, base in zip(self.optimizer.param_groups, self._base_lrs):
            pg["lr"] = warmup_start_lr

    @property
    def in_warmup(self):
        return self._epoch < self.warmup_epochs

    def step(self, val_score=None):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            frac = self._epoch / max(self.warmup_epochs, 1)
            for pg, base in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = self.warmup_start_lr + frac * (base - self.warmup_start_lr)
        else:
            if self.after_scheduler is None:
                return
            if self.after_scheduler_kind == "plateau":
                self.after_scheduler.step(val_score)
            else:
                self.after_scheduler.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=20):
        self.patience = patience
        self.best = None
        self.bad = 0

    def step(self, score):
        if self.best is None or score > self.best:
            self.best = score
            self.bad = 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self):
        return self.bad >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# Split utility
# ─────────────────────────────────────────────────────────────────────────────

def split_by_motion(motion_ids, val_ratio=0.1, seed=0):
    rng = np.random.RandomState(seed)
    motion_ids = np.asarray(motion_ids)
    perm = rng.permutation(len(motion_ids))
    n_val = int(round(len(motion_ids) * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# Validation batch builder / evaluator (retrieval-style)
# ─────────────────────────────────────────────────────────────────────────────

def build_val_batches(
    motion_ids,
    map_df,
    text_lookup,
    val_indices,
    n_batches=100,
    batch_size=32,
    seed=0,
):
    """
    Creates retrieval-style validation batches using ONLY val_indices.

    Returns list of dicts:
      query_text_tensor_id : int
      candidate_motion_rows: np.ndarray shape (batch_size,)
      gt_index             : int (0..batch_size-1)
    """
    rng = np.random.default_rng(seed)

    tid_cols = [c for c in map_df.columns if c.startswith("text_id_")]
    if len(tid_cols) == 0:
        raise ValueError("map_df must contain at least one column named like 'text_id_1'.")

    mid_to_tids: dict[int, list[int]] = {}
    for r in map_df.itertuples(index=False):
        tids = []
        for c in tid_cols:
            tid = getattr(r, c)
            if pd.isna(tid):
                continue
            tids.append(int(text_lookup[tid]))
        if len(tids) > 0:
            mid_to_tids[int(getattr(r, "motion_id"))] = tids

    val_indices = np.array([int(i) for i in val_indices], dtype=np.int64)
    eligible = [int(i) for i in val_indices if int(motion_ids[int(i)]) in mid_to_tids]

    if len(eligible) < batch_size:
        raise ValueError(
            f"Not enough eligible val motions with texts: {len(eligible)} < {batch_size}"
        )

    batches = []
    for _ in range(n_batches):
        pos_row = int(rng.choice(eligible))
        pos_mid = int(motion_ids[pos_row])
        q_tid = int(rng.choice(mid_to_tids[pos_mid]))

        neg_rows: set[int] = set()
        while len(neg_rows) < (batch_size - 1):
            r = int(rng.choice(val_indices))
            if r != pos_row:
                neg_rows.add(r)
        neg_rows_arr = np.array(list(neg_rows), dtype=np.int64)

        cand_rows = np.concatenate([np.array([pos_row], dtype=np.int64), neg_rows_arr])
        perm = rng.permutation(batch_size)
        cand_rows = cand_rows[perm]
        gt_index = int(np.where(perm == 0)[0][0])

        batches.append(
            dict(
                query_text_tensor_id=q_tid,
                candidate_motion_rows=cand_rows,
                gt_index=gt_index,
            )
        )
    return batches


@torch.no_grad()
def eval_val_batches(
    motion_clip_model,
    val_batches,
    motion_paths,
    text_emb,
    device=None,
    ks=(1, 2, 3, 5, 10),
):
    """
    Lazy retrieval-style validation.
    Returns (composite_score, recalls_dict, mrr).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_clip_model = motion_clip_model.to(device).eval()

    recalls = {k: 0 for k in ks}
    mrr_total = 0.0
    n = len(val_batches)

    for b in val_batches:
        q_tid = b["query_text_tensor_id"]
        cand_rows = b["candidate_motion_rows"]
        gt = b["gt_index"]

        q = text_emb[q_tid : q_tid + 1].to(dtype=torch.float32, device=device)  # (1,E)
        q = F.normalize(q, dim=-1)

        motions, lengths = [], []
        D = None
        for r in cand_rows:
            x = np.load(motion_paths[int(r)]).astype(np.float32)
            if D is None:
                D = x.shape[1]
            lengths.append(x.shape[0])
            motions.append(torch.from_numpy(x))

        T_max = max(lengths)
        motion_pad = torch.zeros((len(motions), T_max, D), dtype=torch.float32)
        for i, mv in enumerate(motions):
            motion_pad[i, : mv.shape[0]] = mv
        motion_pad = motion_pad.to(device)

        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        m_emb = F.normalize(motion_clip_model(motion_pad, lengths=lengths_t), dim=-1)  # (B,E)

        sims = (m_emb @ q.t()).squeeze(1)  # (B,)
        ranking = torch.argsort(sims, descending=True).tolist()

        rank_of_gt = ranking.index(gt) + 1  # 1-based
        mrr_total += 1.0 / rank_of_gt

        for k in ks:
            if gt in ranking[:k]:
                recalls[k] += 1

    for k in ks:
        recalls[k] /= max(n, 1)
    mrr = mrr_total / max(n, 1)

    # Composite score (kept for compatibility)
    weights = {k: 1.0 / k for k in ks}
    score = sum(weights[k] * recalls[k] for k in ks) / sum(weights.values())
    return score, recalls, mrr


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler factory
# ─────────────────────────────────────────────────────────────────────────────

def make_scheduler(opt, scheduler_cfg, epochs=None, steps_per_epoch=None):
    if not scheduler_cfg:
        return None, "none"

    cfg = dict(scheduler_cfg)
    name = cfg.pop("name").lower()

    if name in ("plateau", "reducelronplateau"):
        cfg.setdefault("mode", "max")
        cfg.setdefault("factor", 0.5)
        cfg.setdefault("patience", 5)
        return ReduceLROnPlateau(opt, **cfg), "plateau"

    if name in ("step", "steplr"):
        cfg.setdefault("step_size", 10)
        cfg.setdefault("gamma", 0.5)
        return StepLR(opt, **cfg), "epoch"

    if name in ("multistep", "multisteplr"):
        cfg.setdefault("milestones", [30, 60])
        cfg.setdefault("gamma", 0.2)
        return MultiStepLR(opt, **cfg), "epoch"

    if name in ("cosine", "cosineannealinglr"):
        cfg.setdefault("T_max", epochs if epochs is not None else 50)
        cfg.setdefault("eta_min", 1e-6)
        return CosineAnnealingLR(opt, **cfg), "epoch"

    if name in ("cosine_wr", "cosinewarmrestarts", "cosineannealingwarmrestarts"):
        cfg.setdefault("T_0", 30)
        cfg.setdefault("T_mult", 2)
        cfg.setdefault("eta_min", 1e-6)
        return CosineAnnealingWarmRestarts(opt, **cfg), "epoch"

    if name in ("exp", "exponential", "exponentiallr"):
        cfg.setdefault("gamma", 0.98)
        return ExponentialLR(opt, **cfg), "epoch"

    if name in ("onecycle", "onecyclelr"):
        if steps_per_epoch is None or epochs is None:
            raise ValueError("OneCycleLR needs epochs and steps_per_epoch.")
        cfg.setdefault("max_lr", max(pg["lr"] for pg in opt.param_groups))
        cfg.setdefault("total_steps", epochs * steps_per_epoch)
        return OneCycleLR(opt, **cfg), "batch"

    raise ValueError(f"Unknown scheduler name: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Loss (A–C live here + in the epoch loop)
# ─────────────────────────────────────────────────────────────────────────────

def clip_loss(motion_z, text_z, logit_scale, noise_std: float = 0.0):
    """Symmetric CLIP loss with optional embedding noise (train only)."""
    # C) tiny embedding noise (train only)
    if noise_std and noise_std > 0.0:
        motion_z = motion_z + noise_std * torch.randn_like(motion_z)
        text_z   = text_z   + noise_std * torch.randn_like(text_z)

    motion_z = F.normalize(motion_z, dim=-1)
    text_z   = F.normalize(text_z,   dim=-1)

    # B) clamp temperature (exp(logit_scale)) to avoid overly sharp logits
    scale  = logit_scale.exp().clamp(1e-3, 20.0)
    logits = scale * (motion_z @ text_z.t())  # (B,B)
    labels = torch.arange(logits.size(0), device=logits.device)

    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


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
    weight_decay=1e-3,
    n_val_batches=100,
    val_batch_size=32,
    ks=(1, 2, 3, 5, 10),
    patience=50,
    warmup_epochs=5,
    grad_clip=1.0,
    logit_scale_max=np.log(20.0),     # B) clamp logit_scale (log-space)
    freeze_logit_scale_epochs=10,     # A) freeze temperature for first N epochs
    noise_std=1e-3,                   # C) embedding noise (train only)
    time_padding=True,
    augs: dict | None = None,
    scheduler_cfg: dict | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "text_tensor_id" not in text_df.columns:
        text_df = text_df.copy().reset_index(drop=True)
        text_df["text_tensor_id"] = np.arange(len(text_df), dtype=np.int64)
    text_lookup = dict(zip(text_df["text_id"], text_df["text_tensor_id"]))

    train_idx, val_idx = split_by_motion(motion_ids, val_ratio=val_ratio, seed=seed)

    # Dataset (IMPORTANT: your ClipDataset requires indices)
    train_ds = ClipDataset(
        motion_ids=motion_ids,
        motion_paths=motion_paths,
        map_df=map_df,
        text_lookup=text_lookup,
        text_emb=text_emb,
        indices=train_idx,
        augs=augs,
        seed=seed,
    )

    collate_fn = make_collate_fn(pad_collate=time_padding)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model wrapper
    model = MotionClip(motion_model, text_dim=text_emb.shape[1], text_model_name=text_model_name).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched, sched_kind = make_scheduler(opt, scheduler_cfg, epochs=epochs, steps_per_epoch=len(train_loader))
    warmup = LinearWarmupScheduler(
        opt,
        warmup_epochs=warmup_epochs,
        after_scheduler=sched,
        after_scheduler_kind=sched_kind,
        warmup_start_lr=max(lr * 0.05, 1e-7),
    )

    es = EarlyStopping(patience=patience)

    # Build val batches once (stable eval)
    val_batches = build_val_batches(
        motion_ids=motion_ids,
        map_df=map_df,
        text_lookup=text_lookup,
        val_indices=val_idx,
        n_batches=n_val_batches,
        batch_size=val_batch_size,
        seed=seed + 123,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_score = -1.0
    best_info = {}

    for ep in range(1, epochs + 1):
        # A) freeze temperature early
        if hasattr(model, "logit_scale") and isinstance(model.logit_scale, torch.nn.Parameter):
            model.logit_scale.requires_grad_(ep > freeze_logit_scale_epochs)

        model.train()
        running = 0.0
        n_steps = 0

        for batch in train_loader:
            # patched collate returns (motions, lengths, texts)
            if len(batch) == 2:
                motions, texts = batch
                lengths = None
            else:
                motions, lengths, texts = batch

            # move to device
            if isinstance(motions, list):
                motions = [m.to(device, non_blocking=True) for m in motions]
            else:
                motions = motions.to(device, non_blocking=True)

            if lengths is not None:
                lengths = lengths.to(device, non_blocking=True)

            texts = texts.to(device, non_blocking=True)

            # B) clamp logit_scale before forward (avoid NaNs)
            if hasattr(model, "logit_scale"):
                with torch.no_grad():
                    model.logit_scale.nan_to_num_(nan=np.log(1 / 0.07))
                    model.logit_scale.clamp_(0.0, logit_scale_max)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                motion_z = model(motions, lengths=lengths)
                loss = clip_loss(motion_z, texts, model.logit_scale, noise_std=noise_std)

            if not torch.isfinite(loss).all():
                # skip bad step
                continue

            scaler.scale(loss).backward()

            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            # B) clamp after update too
            if hasattr(model, "logit_scale"):
                with torch.no_grad():
                    model.logit_scale.nan_to_num_(nan=np.log(1 / 0.07))
                    model.logit_scale.clamp_(0.0, logit_scale_max)

            running += float(loss.item())
            n_steps += 1

        warmup.step(val_score=None)
        cur_lr = warmup.get_last_lr()[0]
        train_loss = running / max(n_steps, 1)

        # Validation
        model.eval()
        val_score, recalls, mrr = eval_val_batches(
            motion_clip_model=model,
            val_batches=val_batches,
            motion_paths=motion_paths,
            text_emb=text_emb,
            device=device,
            ks=ks,
        )

        rks = " ".join([f"R@{k}={recalls[k]:.3f}" for k in ks])
        print(
            f"epoch {ep:03d}/{epochs} | lr={cur_lr:.2e} | "
            f"train_loss={train_loss:.4f} | val_score={val_score:.4f} | "
            f"MRR={mrr:.4f} | {rks}"
        )

        improved = es.step(val_score)
        if improved:
            best_score = val_score
            best_info = dict(val_score=val_score, MRR=mrr, recalls=recalls, epoch=ep)
            torch.save(model.state_dict(), save_path)
            print(f"Modèle sauvegardé (Nouveau meilleur score : {val_score:.4f})")
            print(f"Found New Best Model! (MRR={mrr:.4f})")
        else:
            print(f"EarlyStopping counter: {es.bad} out of {es.patience}")

        if es.should_stop:
            print(f"Early stopping at epoch {ep}.")
            break

    # Load best checkpoint for return
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model.motion_model, best_info
