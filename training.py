import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from clip_train_dataset import ClipDataset, make_collate_fn
from motion_clip import MotionClip


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='best_model_checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        # Si le nouveau score n'est pas meilleur que le précédent + delta
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Sauvegarde quand le score de validation augmente.'''
        torch.save(model.state_dict(), self.path)
        print(f"Modèle sauvegardé (Nouveau meilleur score : {self.best_score:.4f})")

def split_by_motion(motion_ids, val_ratio=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(motion_ids))
    rng.shuffle(idx)
    n_val = max(1, int(round(val_ratio * len(idx))))
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])
    return train_idx, val_idx

#====================================Validation utils=======================================
def build_val_batches(
    motion_ids,
    map_df,
    text_lookup,
    val_indices,
    n_batches=30,
    batch_size=32,
    seed=0,
):
    """
    Creates retrieval-style validation batches using ONLY val_indices.

    Returns list of dicts:
      query_text_tensor_id: int
      candidate_motion_rows: np.ndarray shape (32,) (indices into motion_ids/motion_paths)
      gt_index: int (0..31)
    """
    rng = np.random.default_rng(seed)

    tid_cols = [c for c in map_df.columns if c.startswith("text_id_")]
    if len(tid_cols) == 0:
        raise ValueError("map_df must contain at least one column named like 'text_id_1'.")

    mid_to_tids = {}
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
        raise ValueError(f"Not enough eligible val motions with texts: {len(eligible)} < {batch_size}")

    batches = []
    for _ in range(n_batches):
        pos_row = int(rng.choice(eligible))
        pos_mid = int(motion_ids[pos_row])
        q_tid = int(rng.choice(mid_to_tids[pos_mid]))

        neg_rows = set()
        while len(neg_rows) < (batch_size - 1):
            r = int(rng.choice(val_indices))
            if r != pos_row:
                neg_rows.add(r)
        neg_rows = np.array(list(neg_rows), dtype=np.int64)

        cand_rows = np.concatenate([np.array([pos_row], dtype=np.int64), neg_rows])
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
def eval_val_batches(motion_clip_model, val_batches, motion_paths, text_emb, device=None, ks=(1, 2, 3, 5, 10)):
    """
    Lazy validation eval:
      - loads only the 32 motions per retrieval batch
      - pads inside the batch
      - keeps text_emb stored on CPU float16; moves only query embedding to GPU
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_clip_model = motion_clip_model.to(device).eval()

    recalls = {k: 0 for k in ks}
    n = len(val_batches)

    for b in val_batches:
        q_tid = b["query_text_tensor_id"]
        cand_rows = b["candidate_motion_rows"]
        gt = b["gt_index"]

        q = text_emb[q_tid:q_tid+1].to(dtype=torch.float32, device=device)  # (1,E)
        q = F.normalize(q, dim=-1)

        motions = []
        lengths = []
        D = None
        for r in cand_rows:
            x = np.load(motion_paths[int(r)]).astype(np.float32)  # (T,D)
            if D is None:
                D = x.shape[1]
            lengths.append(x.shape[0])
            motions.append(torch.from_numpy(x))

        T_max = max(lengths)
        motion_pad = torch.zeros((len(motions), T_max, D), dtype=torch.float32)
        for i, m in enumerate(motions):
            motion_pad[i, : m.shape[0]] = m
        motion_pad = motion_pad.to(device)

        m = F.normalize(motion_clip_model(motion_pad), dim=-1)  # (32,E)
        sims = (m @ q.t()).squeeze(1)                            # (32,)
        ranking = torch.argsort(sims, descending=True)

        for k in ks:
            if gt in ranking[:k].tolist():
                recalls[k] += 1

    for k in ks:
        recalls[k] /= max(n, 1)

    weights = {k: 1.0 / k for k in ks}
    score = sum(weights[k] * recalls[k] for k in ks) / sum(weights.values())
    return score, recalls


#======================================training========================================================

from torch.optim.lr_scheduler import ReduceLROnPlateau

def clip_loss(motion_z, text_z, logit_scale):
    motion_z = F.normalize(motion_z, dim=-1)
    text_z = F.normalize(text_z, dim=-1)
    scale = logit_scale.exp().clamp(1e-3, 100.0)
    logits = scale * (motion_z @ text_z.t())  # (B,B)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


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
    epochs=10,
    batch_size=64,
    lr=1e-4,
    n_val_batches=30,
    ks=(1, 2, 3, 5, 10),
    patience=7,
    time_padding=True,
    augs: dict | None = None,   # NEW
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
        augs=augs,                # NEW
        seed=seed,                # keep deterministic augmentation stream if desired
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
        batch_size=32,
        seed=seed,
    )

    model = MotionClip(
        motion_model=motion_model,
        text_dim=text_emb.shape[1],
        text_model_name=text_model_name,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=0.001,
        path=save_path,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best = {"val_score": -1.0, "state": None}

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
            scaler.step(opt)
            scaler.update()

            tot += float(loss.item())
            n += 1

        train_loss = tot / max(n, 1)

        val_score, recalls = eval_val_batches(
            motion_clip_model=model,
            val_batches=val_batches,
            motion_paths=motion_paths,
            text_emb=text_emb,
            device=device,
            ks=ks,
        )

        r_str = " ".join([f"R@{k}={recalls[k]:.3f}" for k in ks])
        current_lr = opt.param_groups[0]["lr"]
        print(
            f"epoch {ep}/{epochs} | lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | val_score={val_score:.4f} | {r_str}"
        )

        scheduler.step(val_score)
        early_stopping(val_score, model)

        if val_score > best["val_score"]:
            best["val_score"] = val_score
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print("Found New Best Model!")

        if early_stopping.early_stop:
            print(f"Early stopping déclenché à l'époque {ep}. Arrêt de l'entraînement.")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    return model, {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "val_batches": val_batches,
        "best_val_score": best["val_score"],
    }


