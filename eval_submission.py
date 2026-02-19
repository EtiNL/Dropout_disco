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

def make_st_text_encoder_sota(model_name, device, max_seq_len=32):
    """
    Encodeur de texte version SOTA : renvoie les tokens pour la Cross-Attention 
    ET le vecteur global pour le matching final.
    """
    st_model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def text_encoder(texts):
        # output_value=None permet de récupérer 'token_embeddings'
        outputs = st_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_tensor=True,
            output_value=None
        )
        
        tokens = outputs['token_embeddings']
        B, T, E = tokens.shape
        
        # Padding/Truncation SOTA pour la Cross-Attention
        if T > max_seq_len:
            tokens = tokens[:, :max_seq_len, :]
        else:
            tokens = F.pad(tokens, (0, 0, 0, max_seq_len - T))
            
        # Vecteur global (moyenne) pour la similarité finale
        global_emb = tokens.mean(dim=1)
        
        return tokens, global_emb
        
    return text_encoder

@torch.no_grad()
def build_submission_df(
    motion_clip_model,
    test_root="./data/test",
    top_k=10,
    device=None,
    max_seq_len=32
):
    motion_clip_model.eval()

    if device is None:
        device = next(motion_clip_model.parameters()).device

    # Utilisation de l'encodeur SOTA
    text_encoder = make_st_text_encoder_sota(
        motion_clip_model.text_model_name, 
        device=device, 
        max_seq_len=max_seq_len
    )

    rows = []
    colnames = ["query_id"] + [f"candidate_{i}" for i in range(1, top_k + 1)]

    for folder in _list_numbered_subfolders(test_root):
        query_id = int(os.path.basename(folder))

        # Lecture du texte
        txt_files = glob.glob(os.path.join(folder, "*.txt"))
        with open(txt_files[0], "r", encoding="utf-8") as f:
            text = f.read().strip()

        # 1. Extraction SOTA : Tokens (pour le modèle) + Global (pour la similarité)
        q_tokens, q_global = text_encoder([text])
        q_global = F.normalize(q_global, dim=-1) # [1, D]

        # 2. Chargement et Padding des mouvements
        motion_files = sorted(glob.glob(os.path.join(folder, "motion_*.npy")))
        motions, motion_nums = [], []
        for p in motion_files:
            arr = np.load(p)
            motions.append(torch.from_numpy(arr).float())
            m = re.search(r"(\d+)$", os.path.splitext(os.path.basename(p))[0])
            motion_nums.append(int(m.group(1)) if m else p)

        maxT = max(m.shape[0] for m in motions)
        C = motions[0].shape[1]
        padded = [F.pad(m, (0, 0, 0, maxT - m.shape[0])) for m in motions]
        
        motion_batch = torch.stack(padded, dim=0).to(device)
        lengths = torch.tensor([m.shape[0] for m in motions], device=device)

        # 3. Forward SOTA avec Cross-Attention
        # On expand les tokens de la requête pour tout le batch de mouvements (32 candidates)
        q_tokens_expanded = q_tokens.expand(motion_batch.shape[0], -1, -1)
        
        M = motion_clip_model(motion_batch, text_tokens=q_tokens_expanded, lengths=lengths)
        M = F.normalize(M, dim=-1)

        # 4. Calcul de la similarité Cosinus
        sims = (M @ q_global.t()).squeeze(1)  # [32 candidates]
        
        top_idx = torch.topk(sims, k=min(top_k, sims.numel())).indices.cpu().tolist()
        top_candidates = [motion_nums[i] for i in top_idx]

        row = {"query_id": query_id}
        for i in range(top_k):
            row[f"candidate_{i+1}"] = int(top_candidates[i]) if i < len(top_candidates) else None
        rows.append(row)

    df = pd.DataFrame(rows, columns=colnames).sort_values("query_id").reset_index(drop=True)
    return df