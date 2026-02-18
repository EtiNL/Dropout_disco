import torch
import torch.nn as nn
import numpy as np
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = self._build_pe(max_len, d_model)         # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)                         # (L,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                                     # (d/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)                                                # (1,L,d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, d)
        F = x.size(1)
        if F > self.pe.size(1):
            # extend if needed
            self.pe = self._build_pe(F, self.d_model).to(x.device, dtype=x.dtype)
        return x + self.pe[:, :F].to(dtype=x.dtype, device=x.device)


class MDMMotionEncoder(nn.Module):
    def __init__(self, input_dim=384, latent_dim=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos_enc = SinusoidalPositionalEncoding(latent_dim, max_len=2048)

        
        # 2. Embedding temporel (Diffusion timestep)
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=nhead, 
            dim_feedforward=2048, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.ln_final = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, motions, timesteps=None, lengths=None):
        b, f, c = motions.shape

        if timesteps is None:
            timesteps = torch.zeros((b, 1), device=motions.device)

        x = self.input_proj(motions)                         # (B,F,H)
        x = self.pos_enc(x)
        t_emb = self.time_embed(timesteps).unsqueeze(1)      # (B,1,H)
        x = x + t_emb

        key_padding_mask = None
        if lengths is not None:
            # True = padding (à ignorer)
            idx = torch.arange(f, device=motions.device).unsqueeze(0)  # (1,F)
            key_padding_mask = idx >= lengths.unsqueeze(1)             # (B,F) bool

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B,F,H)

        if lengths is None:
            z = x.mean(dim=1)
        else:
            valid = (~key_padding_mask).to(x.dtype)          # (B,F)
            z = (x * valid.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)

        return self.out_proj(self.ln_final(z))