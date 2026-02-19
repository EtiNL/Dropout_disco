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

from clip_train_dataset import ClipDataset, make_collate_fn, compute_stats_safe
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
    text_emb, # [N, 32, 3584]
    device=None,
    ks=(1, 5, 10),
):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_clip_model = motion_clip_model.to(device).eval()
    recalls = {k: 0 for k in ks}
    mrr_total, n = 0.0, len(val_batches)

    for b in val_batches:
        q_tid, cand_rows, gt = b["query_text_tensor_id"], b["candidate_motion_rows"], b["gt_index"]

        # SOTA: Tokens pour le modèle, Global pour la similarité
        text_tokens = text_emb[q_tid:q_tid+1].to(device).float()
        q_global = F.normalize(text_tokens.mean(dim=1), dim=-1)

        motions, lengths = [], []
        for r in cand_rows:
            x = torch.from_numpy(np.load(motion_paths[r])).float().to(device)
            motions.append(x); lengths.append(x.shape[0])

        T_max = max(lengths)
        motion_pad = torch.zeros((len(motions), T_max, motions[0].shape[-1]), device=device)
        for i, m in enumerate(motions): motion_pad[i, :m.shape[0]] = m

        lengths_t = torch.tensor(lengths, device=device)
        # On expand les tokens pour les 32 candidats
        q_tokens_batch = text_tokens.expand(len(motions), -1, -1)
        
        m_emb = F.normalize(motion_clip_model(motion_pad, text_tokens=q_tokens_batch, lengths=lengths_t), dim=-1)
        sims = (m_emb @ q_global.t()).squeeze(1)
        ranking = torch.argsort(sims, descending=True).tolist()

        rank_of_gt = ranking.index(gt) + 1
        mrr_total += 1.0 / rank_of_gt
        for k in ks:
            if gt in ranking[:k]: recalls[k] += 1

    return {k: v/n for k, v in recalls.items()}, mrr_total/n


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

def clip_loss(motion_z, text_tokens, logit_scale, noise_std=0.0):
    # SOTA: Pooling global pour le contraste
    text_z = text_tokens.mean(dim=1)
    
    if noise_std > 0:
        motion_z = motion_z + torch.randn_like(motion_z) * noise_std
        text_z = text_z + torch.randn_like(text_z) * noise_std

    motion_z, text_z = F.normalize(motion_z, dim=-1), F.normalize(text_z, dim=-1)
    scale = logit_scale.exp().clamp(1e-3, 20.0)
    logits = scale * (motion_z @ text_z.t())
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

# ─────────────────────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────────────────────

def train_clip_with_split(
    motion_model, motion_ids, motion_paths, map_df, text_df, text_emb, text_model_name, save_path,
    val_ratio=0.1, seed=0, epochs=150, batch_size=64, lr=1e-4, weight_decay=0.05,
    n_val_batches=100, val_batch_size=32, ks=(1, 5, 10), patience=20, warmup_epochs=10,
    grad_clip=1.0, logit_scale_max=np.log(20.0), freeze_logit_scale_epochs=10,
    noise_std=1e-3, time_padding=True, augs=None, scheduler_cfg=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "text_tensor_id" not in text_df.columns:
        text_df = text_df.copy().reset_index(drop=True)
        text_df["text_tensor_id"] = np.arange(len(text_df), dtype=np.int64)
    text_lookup = dict(zip(text_df["text_id"], text_df["text_tensor_id"]))

    train_idx, val_idx = split_by_motion(motion_ids, val_ratio, seed)
    train_ds = ClipDataset(motion_ids, motion_paths, map_df, text_lookup, text_emb, train_idx, augs=augs, seed=seed)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True, num_workers=2, collate_fn=make_collate_fn(time_padding))

    model = MotionClip(motion_model, text_dim=text_emb.shape[-1], text_model_name=text_model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    sched, kind = make_scheduler(opt, scheduler_cfg, epochs, len(train_loader))
    warmup = LinearWarmupScheduler(opt, warmup_epochs, sched, kind)
    es = EarlyStopping(patience=patience)
    val_batches = build_val_batches(motion_ids, map_df, text_lookup, val_idx, n_val_batches, val_batch_size, seed)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for ep in range(1, epochs + 1):
        if hasattr(model, "logit_scale"): model.logit_scale.requires_grad_(ep > freeze_logit_scale_epochs)
        model.train()
        running, n_steps = 0.0, 0

        for batch in train_loader:
            motions, lengths, texts = batch
            motions = motions.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True) if lengths is not None else None
            texts = texts.to(device, non_blocking=True).float()

            with torch.no_grad():
                model.logit_scale.nan_to_num_(nan=np.log(1/0.07)).clamp_(0.0, logit_scale_max)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                # SOTA: On passe les tokens au modèle
                motion_z = model(motions, text_tokens=texts, lengths=lengths)
                loss = clip_loss(motion_z, texts, model.logit_scale, noise_std)

            if not torch.isfinite(loss): continue

            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            running += loss.item(); n_steps += 1

        warmup.step(); train_loss = running / max(n_steps, 1)
        val_score, recalls, mrr = eval_val_batches(model, val_batches, motion_paths, text_emb, device, ks)
        
        print(f"ep {ep:03d} | lr {opt.param_groups[0]['lr']:.1e} | loss {train_loss:.4f} | val {val_score:.4f} | MRR {mrr:.4f}")
        
        if es.step(val_score):
            torch.save(model.state_dict(), save_path)
            print("Best Saved.")
        if es.should_stop: break

    if os.path.exists(save_path): model.load_state_dict(torch.load(save_path))
    return model.motion_model, {"best_val": es.best}
