import numpy as np
import torch
from sentence_transformers import SentenceTransformer

INSTRUCTION_MODELS = {
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "intfloat/e5-large-v2",
    "intfloat/multilingual-e5-large-instruct",
}

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

    prompt = (
        "Instruct: Given a motion description, retrieve the most similar motion\nQuery: "
        if model_name in INSTRUCTION_MODELS
        else None
    )

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        text_df["description"].tolist(),
        prompt=prompt,
        show_progress_bar=True,
        convert_to_tensor=True,  # avoids numpy CPU round-trip
    )  # torch tensor on device

    # store compact on CPU
    text_emb = emb.detach().to(dtype=torch.float16, device="cpu")
    text_lookup = dict(zip(text_df["text_id"], text_df["text_tensor_id"]))
    return text_df, text_emb, text_lookup