import torch
import torch.nn as nn
from .MDM import MDMMotionEncoder
from .gru import GRUMotionEncoder


class HybridMotionEncoder(nn.Module):
    def __init__(self, d_in, text_out_dim):
        super().__init__()

        self.gru = GRUMotionEncoder(d_in=d_in, hidden_dim=256, out_dim=512)
        self.mdm = MDMMotionEncoder(input_dim=d_in, latent_dim=512, nhead=4, num_layers=3)

        self.fusion_proj = nn.Sequential(
            nn.Linear(512 + 512, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, text_out_dim),
        )

    def forward(self, motions, lengths=None, timesteps=None):
        """
        motions   : (B, T, D) padded float tensor
        lengths   : (B,) long tensor of true frame counts, or None.
                    - MDM receives it on the model's device (for key_padding_mask).
                    - GRU ignores it: infers lengths from zero-padding internally,
                      and pack_padded_sequence requires CPU lengths anyway.
        timesteps : (B, 1) diffusion timesteps for MDM, or None.
        """
        device = next(self.parameters()).device
        motions = motions.to(device)

        # GRU handles lengths internally on CPU — do not pass them.
        z_gru = self.gru(motions)                            # (B, 512)

        # MDM needs lengths on the same device as motions for the padding mask.
        mdm_lengths = lengths.to(device) if lengths is not None else None
        z_mdm = self.mdm(motions, timesteps=timesteps,
                         lengths=mdm_lengths)                # (B, 512)

        z_combined = torch.cat([z_gru, z_mdm], dim=-1)      # (B, 1024)
        return self.fusion_proj(z_combined)                  # (B, text_out_dim)