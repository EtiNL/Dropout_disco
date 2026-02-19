import torch
import torch.nn as nn
from .MDM import MDMMotionEncoder
from .gru import GRUMotionEncoder


class HybridMotionEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        text_out_dim: int,
        # GRU branch
        gru_hidden_dim: int = 512,
        gru_out_dim: int = 512,
        # MDM branch
        mdm_latent_dim: int = 512,
        mdm_nhead: int = 8,
        mdm_num_layers: int = 4,
        mdm_ff_dim: int = 1024,
        mdm_dropout: float = 0.2,
        # Fusion head
        fusion_dropout: float = 0.3,
    ):
        super().__init__()

        self.gru = GRUMotionEncoder(
            d_in=d_in,
            hidden_dim=gru_hidden_dim,
            out_dim=gru_out_dim,
        )
        self.mdm = MDMMotionEncoder(
            input_dim=d_in,
            latent_dim=mdm_latent_dim,
            nhead=mdm_nhead,
            num_layers=mdm_num_layers,
            ff_dim=mdm_ff_dim,
            dropout=mdm_dropout,
        )

        fused_dim = gru_out_dim + mdm_latent_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fused_dim, text_out_dim),
        )

    def forward(self, motions, lengths=None, timesteps=None):
        """
        motions   : (B, T, D) padded float tensor
        lengths   : (B,) long tensor of true frame counts, or None.
                    GRU infers lengths from zero-padding internally (CPU only).
                    MDM receives lengths on the model device for its padding mask.
        timesteps : (B, 1) diffusion timesteps for MDM, or None.
        """
        device = next(self.parameters()).device
        motions = motions.to(device)

        z_gru = self.gru(motions)                            # (B, gru_out_dim)

        mdm_lengths = lengths.to(device) if lengths is not None else None
        z_mdm = self.mdm(motions, timesteps=timesteps,
                         lengths=mdm_lengths)                # (B, mdm_latent_dim)

        z_combined = torch.cat([z_gru, z_mdm], dim=-1)      # (B, fused_dim)
        return self.fusion_proj(z_combined)                  # (B, text_out_dim)