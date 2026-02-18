import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def zero_module(module):
    """Initialise les poids à zéro pour stabiliser le début de l'entraînement."""
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class StylizationBlock(nn.Module):
    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, h, emb):
        emb_out = self.emb_layers(emb).unsqueeze(1)
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h

class LinearTemporalSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, src_mask):
        B, T, D = x.shape
        H = self.num_head
        query = self.query(self.norm(x))
        # Utilisation du masque pour que l'attention ignore le padding
        key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        return x + self.proj_out(y, emb)

class LinearTemporalCrossAttention(nn.Module):
    def __init__(self, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb):
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        query = self.query(self.norm(x))
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        return x + self.proj_out(y, emb)

class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x + self.proj_out(y, emb)

class DecoderLayer(nn.Module):
    def __init__(self, latent_dim, text_latent_dim, time_embed_dim, ffn_dim, num_head, dropout):
        super().__init__()
        self.sa_block = LinearTemporalSelfAttention(latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = LinearTemporalCrossAttention(latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask):
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb)
        x = self.ffn(x, emb)
        return x

class DiffusionTransformerEncoder(nn.Module):
    def __init__(self, input_feats=384, latent_dim=512, num_layers=8, num_heads=8, ff_size=1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_embed = nn.Linear(input_feats, latent_dim)
        self.register_buffer('pos_encoding', self._get_sinusoidal_encoding(5000, latent_dim))
        
        self.time_embed_dim = latent_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.blocks = nn.ModuleList([
            DecoderLayer(latent_dim, 512, self.time_embed_dim, ff_size, num_heads, 0.1) 
            for _ in range(num_layers)
        ])
        
        self.out_ln = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, 512)

    def _get_sinusoidal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, text_emb=None, lengths=None):
        B, T, _ = x.shape
        device = x.device

        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=device)
        
        grid = torch.arange(T, device=device).unsqueeze(0) # (1, T)
        src_mask = (grid < lengths.unsqueeze(1)).float().unsqueeze(-1) # (B, T, 1)
        
        timesteps = torch.zeros(B, device=device).long()
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
        
        h = self.joint_embed(x)
        h = h + self.pos_encoding[:, :T, :]
        
        for block in self.blocks:
            xf = text_emb.unsqueeze(1) if text_emb is not None else h
            h = block(h, xf, emb, src_mask)
        
        h = h * src_mask 
        z = h.sum(dim=1) / lengths.unsqueeze(1).float()
        
        return self.out_proj(self.out_ln(z))