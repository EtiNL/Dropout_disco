import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class StylizationBlock(nn.Module):
    def __init__(self, latent_dim, time_embed_dim):
        super().__init__()
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 2 * latent_dim))
        nn.init.zeros_(self.emb_layers[-1].weight)
        nn.init.zeros_(self.emb_layers[-1].bias)

    def forward(self, h, emb):
        emb_out = self.emb_layers(emb).unsqueeze(1)
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        return h * (1 + scale) + shift

class CrossAttentionLayer(nn.Module):
    def __init__(self, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(text_latent_dim, latent_dim)
        self.v = nn.Linear(text_latent_dim, latent_dim)
        self.mod = StylizationBlock(latent_dim, time_embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xf, emb):
        B, T, D = x.shape
        q = self.q(self.norm(x)).view(B, T, self.num_head, -1).transpose(1, 2)
        k = self.k(self.text_norm(xf)).view(B, -1, self.num_head, -1).transpose(1, 2)
        v = self.v(self.text_norm(xf)).view(B, -1, self.num_head, -1).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (D**-0.5)
        attn = F.softmax(attn, dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return x + self.mod(self.dropout(h), emb)

class SotaMotionEncoder(nn.Module):
    def __init__(self, input_feats=384, latent_dim=512, text_dim=3584, num_layers=6, nhead=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_embed = nn.Linear(input_feats, latent_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 5000, latent_dim) * 0.02)
        self.time_dim = latent_dim * 4
        self.time_mlp = nn.Sequential(nn.Linear(latent_dim, self.time_dim), nn.SiLU(), nn.Linear(self.time_dim, self.time_dim))
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'sa': nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim*4, batch_first=True, activation='gelu'),
                'ca': CrossAttentionLayer(latent_dim, text_dim, nhead, 0.1, self.time_dim),
                'ff': nn.Sequential(nn.Linear(latent_dim, latent_dim*4), nn.GELU(), nn.Dropout(0.1), nn.Linear(latent_dim*4, latent_dim))
            }) for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(latent_dim)
        self.projection_head = nn.Sequential(nn.Linear(latent_dim, latent_dim*2), nn.SiLU(), nn.Linear(latent_dim*2, text_dim))

    def forward(self, x, text_tokens, lengths=None):
        B, T, _ = x.shape
        if lengths is None: lengths = torch.full((B,), T, device=x.device)
        mask = (torch.arange(T, device=x.device)[None, :] < lengths[:, None]).unsqueeze(-1).float()
        h = self.joint_embed(x) + self.pos_embed[:, :T, :]
        t_emb = self.time_mlp(timestep_embedding(torch.zeros(B, device=x.device).long(), self.latent_dim))

        for layer in self.layers:
            h = layer['sa'](h)
            h = layer['ca'](h, text_tokens, t_emb)
            h = h + layer['ff'](h) # Simplified FF for clarity
        
        h = h * mask
        z = h.sum(dim=1) / lengths.unsqueeze(1).float()
        return self.projection_head(self.out_ln(z))