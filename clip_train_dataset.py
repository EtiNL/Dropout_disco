import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_augmentation import aug_rotate_y, aug_translate_xz

AUG_REGISTRY = {
    "translate_xz": aug_translate_xz,
    "rotate_y": aug_rotate_y,
}

def compute_stats_safe(motion_paths):
    sum_x = np.zeros(384)
    sum_sq_x = np.zeros(384)
    total_frames = 0
    
    for path in motion_paths:
        motion = np.load(path)  
        sum_x += np.sum(motion, axis=0)
        sum_sq_x += np.sum(motion**2, axis=0)
        total_frames += motion.shape[0]
    
    mean = sum_x / total_frames
    var = (sum_sq_x / total_frames) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-8))
    
    std[std < 1e-5] = 1.0  
    return mean, std

class ClipDataset(Dataset):
    """
    Returns (motion_tensor(T,D), text_embedding(E))

    Notes:
      - map_df can have any number of columns named text_id_1, text_id_2, ..., text_id_K.
      - For each motion, one of its available text embeddings is sampled uniformly at random.
    """

    def __init__(
        self,
        motion_ids,
        motion_paths,
        map_df,
        text_lookup,
        text_emb,
        indices,
        mean, 
        std,
        augs: dict | None = None,
        seed: int | None = None,
    ):
        self.motion_ids = motion_ids
        self.motion_paths = motion_paths
        self.text_emb = text_emb  # CPU float16
        self.augs = augs or {}
        self.rng = np.random.default_rng(seed)
        
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

        tid_cols = [c for c in map_df.columns if c.startswith("text_id_")]
        if len(tid_cols) == 0:
            raise ValueError("map_df must contain at least one column named like 'text_id_1'.")

        # motion_id -> list[text_tensor_id]
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

        self.items = []
        for i in indices:
            mid = int(motion_ids[i])
            tids = mid_to_tids.get(mid, [])
            if len(tids) > 0:
                self.items.append((int(i), tids))

        if len(self.items) == 0:
            raise ValueError("No (motion,text) training samples in this split.")

    def __len__(self):
        return len(self.items)

    def _apply_augs(self, x: np.ndarray) -> np.ndarray:
        for name, kwargs in self.augs.items():
            fn = AUG_REGISTRY.get(name)
            if fn is None:
                raise ValueError(f"Unknown augmentation '{name}'. Available: {list(AUG_REGISTRY.keys())}")
            if kwargs is None:
                kwargs = {}
            x = fn(x, rng=self.rng, **kwargs)
        return x

    def __getitem__(self, k):
        row_i, tids = self.items[k]
        path = self.motion_paths[row_i]

        x = np.load(path).astype(np.float32)  # (T,D)

        # apply augmentations (NumPy-side)
        x = self._apply_augs(x)
        
        x = torch.from_numpy(x)
        x = (x - self.mean) / self.std

        tid = tids[self.rng.integers(len(tids))]
        tvec = self.text_emb[tid].to(dtype=torch.float32)

        return x, tvec


def make_collate_fn(pad_collate: bool = True):

    def _pad(batch):
        motions, texts = zip(*batch)
        lengths = torch.tensor([m.shape[0] for m in motions], dtype=torch.long)  # (B,)
        T_max = int(lengths.max().item())
        D = motions[0].shape[1]

        motion_pad = torch.zeros((len(motions), T_max, D), dtype=torch.float32)
        for i, m in enumerate(motions):
            motion_pad[i, : m.shape[0]] = m

        texts_t = torch.stack(texts, dim=0)
        return motion_pad, lengths, texts_t

    def _nopad(batch):
        motions, texts = zip(*batch)
        lengths = torch.tensor([m.shape[0] for m in motions], dtype=torch.long)  # (B,)
        motions = [m.to(dtype=torch.float32) for m in motions]
        texts_t = torch.stack(texts, dim=0)
        return motions, lengths, texts_t

    return _pad if pad_collate else _nopad

