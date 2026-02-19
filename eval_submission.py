import os
import glob
import re

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer


def _list_numbered_subfolders(root):
    subs = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and re.fullmatch(r"\d+", name):
            subs.append((int(name), p))
    return [p for _, p in sorted(subs, key=lambda x: x[0])]

def make_st_text_encoder(model_name, device):
    st_model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def text_encoder(texts):
        return st_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
    return text_encoder

@torch.no_grad()
def build_submission_df(
    motion_clip_model,
    test_root="./data/test",
    top_k=10,
    device=None,
):
    """
    Returns a DF like:
      query_id | candidate_1 | ... | candidate_10
    where each test_root/<query_id>/ contains:
      - 1 text file (*.txt)
      - 32 motions: motion_<j>.npy for j=1..32
    and candidates are the motion indices j (ints) ranked by similarity.
    """

    motion_clip_model.eval()

    if device is None:
        try:
            device = next(motion_clip_model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not hasattr(motion_clip_model, "text_model_name"):
        raise AttributeError("motion_clip_model must have attribute text_model_name.")

    text_encoder = make_st_text_encoder(motion_clip_model.text_model_name, device=device)
    encode_motion = getattr(motion_clip_model, "encode_motion", None)

    rows = []
    colnames = ["query_id"] + [f"candidate_{i}" for i in range(1, top_k + 1)]

    for folder in _list_numbered_subfolders(test_root):
        query_id = int(os.path.basename(folder))

        txt_files = sorted(glob.glob(os.path.join(folder, "*.txt")))
        if len(txt_files) != 1:
            raise FileNotFoundError(f"{folder}: expected exactly 1 .txt, found {len(txt_files)}")

        motion_files = sorted(glob.glob(os.path.join(folder, "motion_*.npy")))
        if len(motion_files) == 0:
            # fallback if names differ
            motion_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        if len(motion_files) == 0:
            raise FileNotFoundError(f"{folder}: no .npy found")

        # read text
        with open(txt_files[0], "r", encoding="utf-8") as f:
            text = f.read().strip()

        # text embedding (fp32)
        q = text_encoder([text])
        q = F.normalize(q, dim=-1).to(device=device, dtype=torch.float32)  # [1, D]

        # load motions
        motions = []
        motion_nums = []
        for p in motion_files:
            arr = np.load(p)
            t = torch.from_numpy(arr)

            if t.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                t = t.float()
            else:
                t = t.to(dtype=torch.float32)

            motions.append(t)

            base = os.path.splitext(os.path.basename(p))[0]  # e.g. motion_17
            m = re.search(r"(\d+)$", base)
            motion_nums.append(int(m.group(1)) if m else base)

        # record true lengths before any padding
        lengths = torch.tensor([m.shape[0] for m in motions], dtype=torch.long)

        # stack or pad (padding assumes [T, C] if variable length)
        try:
            motion_batch = torch.stack(motions, dim=0)
        except RuntimeError:
            if motions[0].ndim != 2:
                raise ValueError(f"{folder}: padding assumes motions are [T, C]. Adapt if different.")
            maxT = max(m.shape[0] for m in motions)
            C = motions[0].shape[1]
            padded = []
            for m in motions:
                padT = maxT - m.shape[0]
                if padT > 0:
                    m = torch.cat([m, torch.zeros(padT, C, dtype=m.dtype)], dim=0)
                padded.append(m)
            motion_batch = torch.stack(padded, dim=0)

        motion_batch = motion_batch.to(device=device, dtype=torch.float32)
        # lengths stays on CPU: pack_padded_sequence (used by GRU) requires CPU lengths.
        # HybridMotionEncoder moves lengths to the right device internally for MDM.

        # motion embeddings (fp32)
        if encode_motion is not None:
            M = encode_motion(motion_batch)
        else:
            M = motion_clip_model(motion_batch, lengths=lengths)

        if M.ndim == 1:
            M = M.unsqueeze(0)

        M = F.normalize(M, dim=-1).to(device=device, dtype=torch.float32)
        q = q.to(device=device, dtype=torch.float32)

        # similarity + topk
        sims = (M @ q.t()).squeeze(1)  # [Nm]
        k = min(top_k, sims.numel())
        top_idx = torch.topk(sims, k=k, largest=True).indices.cpu().tolist()

        top_candidates = [motion_nums[i] for i in top_idx]
        row = {"query_id": query_id}
        for i in range(top_k):
            row[f"candidate_{i+1}"] = int(top_candidates[i]) if i < len(top_candidates) else None

        rows.append(row)

    df = pd.DataFrame(rows, columns=colnames).sort_values("query_id").reset_index(drop=True)
    return df