import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

INSTRUCTION_MODELS = {
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "intfloat/e5-large-v2",
    "intfloat/multilingual-e5-large-instruct",
}

def embed_texts(text_df, model_name="Qwen/Qwen3-Embedding-0.6B", max_seq_len=32):
    """
    Modifié pour l'architecture SOTA (Cross-Attention) :
    Returns:
      text_df      : copy + column text_tensor_id
      text_emb     : torch.Tensor [N_texts, max_seq_len, E] (Float16 sur CPU)
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
    
    outputs = model.encode(
        text_df["description"].tolist(),
        prompt=prompt,
        show_progress_bar=True,
        convert_to_tensor=True,
        output_value=None 
    )

    raw_tokens = outputs['token_embeddings']
    B, T, E = raw_tokens.shape

    # Pour la Cross-Attention, on veut une forme fixe (B, max_seq_len, E)
    if T > max_seq_len:
        # On coupe si c'est trop long
        text_emb = raw_tokens[:, :max_seq_len, :]
    else:
        # On ajoute des zéros si c'est trop court
        padding_size = max_seq_len - T
        # F.pad pad sur les dernières dimensions : (dimE_gauche, dimE_droite, dimT_gauche, dimT_droite)
        text_emb = F.pad(raw_tokens, (0, 0, 0, padding_size))

    text_emb = text_emb.detach().to(dtype=torch.float16, device="cpu")
    text_lookup = dict(zip(text_df["text_id"], text_df["text_tensor_id"]))
    
    print(f"Encodage SOTA terminé. Forme du tenseur : {text_emb.shape}")
    return text_df, text_emb, text_lookup