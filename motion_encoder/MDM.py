import math
import numpy as np
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = self._build_pe(max_len, d_model)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1,L,d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, d)
        F = x.size(1)
        if F > self.pe.size(1):
            pe = self._build_pe(F, self.d_model).to(x.device, dtype=x.dtype)
            self.pe = pe
        return x + self.pe[:, :F].to(dtype=x.dtype, device=x.device)


class MDMMotionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        latent_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        max_len: int = 2048,
        dropout: float = 0.1,
        ff_dim: int = 2048,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Positional encoding (sinusoidal) + gentle injection
        self.pos_enc = SinusoidalPositionalEncoding(latent_dim, max_len=max_len)
        self.pos_scale = nn.Parameter(torch.tensor(0.1))
        self.pos_drop = nn.Dropout(dropout)

        # CLS token for sequence readout
        self.cls = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.drop = nn.Dropout(dropout)

        # Diffusion timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            activation="gelu",
            batch_first=True,
            dropout=dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln_final = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        # Init CLS token to small values (stable)
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, motions: torch.Tensor, timesteps=None, lengths=None) -> torch.Tensor:
        """
        motions:  (B, F, D)
        timesteps:(B, 1) or None
        lengths:  (B,) long tensor giving true lengths in [1..F] when padded; optional
        """
        b, f, _ = motions.shape

        if timesteps is None:
            timesteps = torch.zeros((b, 1), device=motions.device, dtype=motions.dtype)
        else:
            timesteps = timesteps.to(device=motions.device, dtype=motions.dtype)

        x = self.input_proj(motions)  # (B, F, H)

        # Gentle PE: x + alpha * PE, plus dropout
        # Use the buffer directly to avoid re-adding PE twice.
        if f > self.pos_enc.pe.size(1):
            _ = self.pos_enc(x)  # extends internal buffer if needed
        pe = self.pos_enc.pe[:, :f].to(device=x.device, dtype=x.dtype)
        x = x + self.pos_scale * pe
        x = self.pos_drop(x)

        # Add diffusion timestep embedding (broadcast over frames)
        t_emb = self.time_embed(timesteps).unsqueeze(1)  # (B, 1, H)
        x = x + t_emb

        # Prepend CLS
        cls = self.cls.expand(b, -1, -1).to(device=x.device, dtype=x.dtype)  # (B,1,H)
        x = torch.cat([cls, x], dim=1)  # (B, 1+F, H)
        x = self.drop(x)

        # Padding mask (True = ignore)
        key_padding_mask = None
        if lengths is not None:
            lengths = lengths.to(device=x.device)
            idx = torch.arange(f, device=x.device).unsqueeze(0)  # (1,F)
            pad = idx >= lengths.unsqueeze(1)  # (B,F) True=pad
            cls_pad = torch.zeros((b, 1), dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_pad, pad], dim=1)  # (B,1+F)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B,1+F,H)

        z = x[:, 0]  # CLS readout
        return self.out_proj(self.ln_final(z))
