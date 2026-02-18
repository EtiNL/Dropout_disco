import torch
import torch.nn as nn
import numpy as np

class MDMMotionEncoder(nn.Module):
    def __init__(self, input_dim=384, latent_dim=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
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

    def forward(self, motions, timesteps=None):
        # motions shape: (Batch, Frames, 384)
        b, f, c = motions.shape
        
        if timesteps is None:
            timesteps = torch.zeros((b, 1), device=motions.device)
            
        x = self.input_proj(motions) # (B, F, 512)
        t_emb = self.time_embed(timesteps).unsqueeze(1) # (B, 1, 512)
        
        x = x + t_emb # On injecte le niveau de bruit/temps
        
        
        x = self.transformer(x) 
        
        z = x.mean(dim=1) 
        
        return self.out_proj(self.ln_final(z))