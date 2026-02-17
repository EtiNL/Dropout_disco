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
    def __init__(self, patience=50, min_delta=0.001, path="best_model_checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self._save(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self._save(model)
            self.counter = 0

    def _save(self, model):
        torch.save(model.state_dict(), self.path)
        print(f"Modèle sauvegardé (Nouveau meilleur score : {self.best_score:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Train / val split
# ─────────────────────────────────────────────────────────────────────────────

def split_by_motion(motion_ids, val_ratio=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(motion_ids))
    rng.shuffle(idx)
    n_val = max(1, int(round(val_ratio * len(idx))))
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])
    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# Validation utils
# ─────────────────────────────────────────────────────────────────────────────

def build_val_batches(
    motion_ids,
    map_df,
    text_lookup,
    val_indices,
    n_batches=100,          # FIX: was 30 – far too noisy; use ≥100
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

    composite_score: weighted average of recalls with 1/k weights (original metric).
    mrr            : Mean Reciprocal Rank – more stable and informative.
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

        m_emb = F.normalize(motion_clip_model(motion_pad), dim=-1)  # (B,E)
        sims = (m_emb @ q.t()).squeeze(1)                            # (B,)
        ranking = torch.argsort(sims, descending=True).tolist()

        rank_of_gt = ranking.index(gt) + 1  # 1-based
        mrr_total += 1.0 / rank_of_gt

        for k in ks:
            if gt in ranking[:k]:
                recalls[k] += 1

    for k in ks:
        recalls[k] /= max(n, 1)
    mrr = mrr_total / max(n, 1)

    # Composite score (original, kept for compatibility)
    weights = {k: 1.0 / k for k in ks}
    score = sum(weights[k] * recalls[k] for k in ks) / sum(weights.values())
    return score, recalls, mrr


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler factory
# ─────────────────────────────────────────────────────────────────────────────

def make_scheduler(opt, scheduler_cfg, epochs=None, steps_per_epoch=None):
    """
    scheduler_cfg examples:
      None -> no scheduler
      {"name":"plateau",    "mode":"max", "factor":0.5, "patience":5}
      {"name":"step",       "step_size":10, "gamma":0.5}
      {"name":"multistep",  "milestones":[30,60], "gamma":0.2}
      {"name":"cosine",     "T_max":50, "eta_min":1e-6}
      {"name":"cosine_wr",  "T_0":30, "T_mult":2, "eta_min":1e-6}   # ← recommended
      {"name":"exp",        "gamma":0.98}
      {"name":"onecycle",   "max_lr":1e-3, "pct_start":0.1}

    Returns (scheduler, step_kind) where step_kind ∈ {"none","plateau","epoch","batch"}.
    """
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
        cfg.setdefault("gamma", 0.1)
        return StepLR(opt, **cfg), "epoch"

    if name in ("multistep", "multisteplr"):
        cfg.setdefault("milestones", [30, 60])
        cfg.setdefault("gamma", 0.2)
        return MultiStepLR(opt, **cfg), "epoch"

    if name in ("cosine", "cosineannealinglr"):
        if epochs is None:
            raise ValueError("cosine scheduler needs `epochs`.")
        cfg.setdefault("T_max", epochs)
        cfg.setdefault("eta_min", 1e-6)
        return CosineAnnealingLR(opt, **cfg), "epoch"

    if name in ("cosine_wr", "cosineannealingwarmrestarts"):
        cfg.setdefault("T_0", 30)
        cfg.setdefault("T_mult", 2)
        cfg.setdefault("eta_min", 1e-6)
        return CosineAnnealingWarmRestarts(opt, **cfg), "epoch"

    if name in ("exp", "exponentiallr"):
        cfg.setdefault("gamma", 0.99)
        return ExponentialLR(opt, **cfg), "epoch"

    if name in ("onecycle", "onecyclelr"):
        if epochs is None or steps_per_epoch is None:
            raise ValueError("onecycle needs `epochs` and `steps_per_epoch`.")
        cfg.setdefault("max_lr", opt.param_groups[0]["lr"])
        return OneCycleLR(opt, epochs=epochs, steps_per_epoch=steps_per_epoch, **cfg), "batch"

    raise ValueError(f"Unknown scheduler name: '{name}'")


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def clip_loss(motion_z, text_z, logit_scale):
    motion_z = F.normalize(motion_z, dim=-1)
    text_z   = F.normalize(text_z,   dim=-1)
    scale    = logit_scale.exp().clamp(1e-3, 100.0)
    logits   = scale * (motion_z @ text_z.t())          # (B, B)
    labels   = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
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
    logit_scale_max=np.log(100.0),  # NEW: clamp logit_scale after each step
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
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    val_batches = build_val_batches(
        motion_ids=motion_ids,
        map_df=map_df,
        text_lookup=text_lookup,
        val_indices=val_idx,
        n_batches=n_val_batches,
        batch_size=val_batch_size,
        seed=seed,
    )

    model = MotionClip(
        motion_model=motion_model,
        text_dim=text_emb.shape[1],
        text_model_name=text_model_name,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Default scheduler: cosine warm restarts (better than plain cosine for long runs)
    if scheduler_cfg is None:
        scheduler_cfg = {"name": "cosine_wr", "T_0": 30, "T_mult": 2, "eta_min": 1e-6}

    base_scheduler, sched_step = make_scheduler(
        opt,
        scheduler_cfg,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    # Wrap with linear warmup (only for epoch/plateau schedulers, not onecycle)
    if warmup_epochs > 0 and sched_step != "batch":
        scheduler = LinearWarmupScheduler(
            opt,
            warmup_epochs=warmup_epochs,
            after_scheduler=base_scheduler,
            after_scheduler_kind=sched_step,
        )
        effective_sched_step = "warmup"
    else:
        scheduler = base_scheduler
        effective_sched_step = sched_step

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=0.001,
        path=save_path,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best = {"val_score": -1.0, "mrr": -1.0, "state": None}

    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0

        for motions, texts in train_loader:
            if time_padding:
                motions = motions.to(device, non_blocking=True)
            else:
                motions = [m.to(device, non_blocking=True) for m in motions]
            texts = texts.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                motion_z = model(motions)
                loss = clip_loss(motion_z, texts, model.logit_scale)

            scaler.scale(loss).backward()

            # FIX: gradient clipping (unscale first so clip operates on true grads)
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(opt)
            scaler.update()

            # FIX: clamp logit_scale so temperature doesn't collapse
            with torch.no_grad():
                model.logit_scale.clamp_(max=logit_scale_max)

            if effective_sched_step == "batch":
                scheduler.step()

            tot += float(loss.item())
            n += 1

        train_loss = tot / max(n, 1)

        val_score, recalls, mrr = eval_val_batches(
            motion_clip_model=model,
            val_batches=val_batches,
            motion_paths=motion_paths,
            text_emb=text_emb,
            device=device,
            ks=ks,
        )

        # Scheduler step
        if effective_sched_step == "warmup":
            scheduler.step(val_score=val_score if sched_step == "plateau" else None)
        elif effective_sched_step == "plateau":
            scheduler.step(val_score)
        elif effective_sched_step == "epoch":
            scheduler.step()

        current_lr = opt.param_groups[0]["lr"]
        r_str = " ".join([f"R@{k}={recalls[k]:.3f}" for k in ks])
        print(
            f"epoch {ep:03d}/{epochs:03d} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | val_score={val_score:.4f} | "
            f"MRR={mrr:.4f} | {r_str}"
        )

        # Track best by MRR (more stable) but also keep composite score for compat
        early_stopping(val_score, model)

        if float(val_score) > best["val_score"]:
            best["val_score"] = float(val_score)
            best["mrr"] = float(mrr)
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"Found New Best Model! (MRR={mrr:.4f})")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {ep}.")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    return model, {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "val_batches": val_batches,
        "best_val_score": best["val_score"],
        "best_mrr": best["mrr"],
        "scheduler_cfg": scheduler_cfg,
    }