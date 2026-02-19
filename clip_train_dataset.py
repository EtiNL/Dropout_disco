import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_augmentation import aug_rotate_y, aug_translate_xz, aug_swap_persons

# On ajoute le swap_persons au registre pour le SOTA
AUG_REGISTRY = {
    "translate_xz": aug_translate_xz,
    "rotate_y": aug_rotate_y,
    "swap_persons": aug_swap_persons, 
}

class ClipDataset(Dataset):
    def __init__(
        self,
        motion_ids,
        motion_paths,
        map_df,
        text_lookup,
        text_emb,
        indices,
        mean=None, # Recommandé pour SOTA
        std=None,  # Recommandé pour SOTA
        augs: dict | None = None,
        seed: int | None = None,
    ):
        self.motion_ids = motion_ids
        self.motion_paths = motion_paths
        self.text_emb = text_emb
        self.text_lookup = text_lookup # Ajouté pour la cohérence
        self.augs = augs or {}
        self.rng = np.random.default_rng(seed)
        
        # Stockage des stats de normalisation sur CPU
        self.mean = torch.tensor(mean).float() if mean is not None else None
        self.std = torch.tensor(std).float() if std is not None else None

        tid_cols = [c for c in map_df.columns if c.startswith("text_id_")]
        
        # Mapping Motion -> List of Text IDs
        mid_to_tids = {}
        for r in map_df.itertuples(index=False):
            tids = []
            for c in tid_cols:
                tid = getattr(r, c)
                if pd.isna(tid) or tid not in text_lookup:
                    continue
                tids.append(int(text_lookup[tid]))
            if len(tids) > 0:
                mid_to_tids[int(getattr(r, "motion_id"))] = tids

        self.items = []
        for i in indices:
            mid = int(motion_ids[i])
            if mid in mid_to_tids:
                self.items.append((int(i), mid_to_tids[mid]))

    def __len__(self):
        return len(self.items)

    def _apply_augs(self, x: np.ndarray) -> np.ndarray:
        for name, kwargs in self.augs.items():
            fn = AUG_REGISTRY.get(name)
            if fn:
                x = fn(x, rng=self.rng, **(kwargs or {}))
        return x

    def __getitem__(self, k):
        row_i, tids = self.items[k]
        path = self.motion_paths[row_i]
        
        # 1. Load
        x = np.load(path).astype(np.float32)
        
        # 2. Augment (NumPy)
        x = self._apply_augs(x)
        x = torch.from_numpy(x)

        # 3. Normalize (Crucial pour les Transformers)
        if self.mean is not None:
            x = (x - self.mean) / self.std

        # 4. Text tokens
        tid = tids[self.rng.integers(len(tids))]
        t_tokens = self.text_emb[tid].to(dtype=torch.float32) # [30, 3584]

        return x, t_tokens

def make_collate_fn(pad_collate: bool = True):
    def _pad(batch):
        motions, texts = zip(*batch)
        lengths = torch.tensor([m.shape[0] for m in motions], dtype=torch.long)
        T_max = int(lengths.max().item())
        D = motions[0].shape[1]

        # Padding des motions
        motion_pad = torch.zeros((len(motions), T_max, D), dtype=torch.float32)
        for i, m in enumerate(motions):
            motion_pad[i, : m.shape[0]] = m

        # Stack des tokens (longueur fixe déjà gérée à l'encodage)
        texts_t = torch.stack(texts, dim=0) 
        
        return motion_pad, lengths, texts_t

    return _pad if pad_collate else None # Simplifié pour ton usage