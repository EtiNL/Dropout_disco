import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class ClipDataset(Dataset):
    """
    Training dataset:
      returns (motion_tensor(T,D), text_embedding(E))
    Motions are loaded lazily from disk (no huge RAM tensor).
    """
    def __init__(self, motion_ids, motion_paths, map_df, text_lookup, text_emb, indices):
        self.motion_ids = motion_ids
        self.motion_paths = motion_paths
        self.text_emb = text_emb  # CPU float16

        # motion_id -> list[text_tensor_id]
        mid_to_tids = {}
        for r in map_df.itertuples(index=False):
            tids = []
            for tid in (r.text_id_1, r.text_id_2, r.text_id_3):
                if pd.isna(tid):
                    continue
                tids.append(int(text_lookup[tid]))
            if len(tids) > 0:
                mid_to_tids[int(r.motion_id)] = tids

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

    def __getitem__(self, k):
        row_i, tids = self.items[k]
        path = self.motion_paths[row_i]

        x = np.load(path).astype(np.float32)     # (T,D), loaded on demand
        tid = tids[np.random.randint(len(tids))]

        # use fp32 in loss for stability
        tvec = self.text_emb[tid].to(dtype=torch.float32)

        return torch.from_numpy(x), tvec


def make_collate_fn(pad_collate: bool = True):
    """
    If pad_collate=True:
      motions are padded to (B, T_max, D) and returned as float32 tensor.

    If pad_collate=False:
      motions are returned as a Python list of (T_i, D) float32 tensors
      (variable-length), and texts as (B, E) float32 tensor.
      Use this if your model can handle variable-length sequences without padding.
    """
    def _pad(batch):
        motions, texts = zip(*batch)
        lengths = [m.shape[0] for m in motions]
        T_max = max(lengths)
        D = motions[0].shape[1]

        motion_pad = torch.zeros((len(motions), T_max, D), dtype=torch.float32)
        for i, m in enumerate(motions):
            motion_pad[i, : m.shape[0]] = m

        texts_t = torch.stack(texts, dim=0)
        return motion_pad, texts_t

    def _nopad(batch):
        motions, texts = zip(*batch)
        # ensure float32 motions (already are, but keep invariant)
        motions = [m.to(dtype=torch.float32) for m in motions]
        texts_t = torch.stack(texts, dim=0)
        return motions, texts_t

    return _pad if pad_collate else _nopad
