import numpy as np

import torch
from sentence_transformers import SentenceTransformer


def embed_texts(text_df, model_name="Qwen/Qwen3-Embedding-0.6B"):
    """
    Returns:
      text_df      : copy + column text_tensor_id
      text_emb     : torch.Tensor on CPU, dtype=float16, shape (N_texts, E)
      text_lookup  : dict {text_id -> text_tensor_id}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_df = text_df.copy().reset_index(drop=True)
    text_df["text_tensor_id"] = np.arange(len(text_df), dtype=np.int64)

    model = SentenceTransformer(model_name, device=device)

    emb = model.encode(
        text_df["description"].tolist(),
        show_progress_bar=True,
        convert_to_tensor=True,   # avoids numpy CPU round-trip
    )  # torch tensor on device

    # store compact on CPU
    text_emb = emb.detach().to(dtype=torch.float16, device="cpu")

    text_lookup = dict(zip(text_df["text_id"], text_df["text_tensor_id"]))
    return text_df, text_emb, text_lookup


