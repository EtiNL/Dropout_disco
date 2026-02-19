import torch
import torch.nn as nn
from motion_encoder.MDM import MDMMotionEncoder
from motion_encoder.gru import GRUMotionEncoder


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
                    Passed to MDMMotionEncoder so its Transformer ignores
                    padding frames. GRUMotionEncoder infers lengths internally
                    from zero-padding, so it does not receive this argument.
        timesteps : (B, 1) diffusion timesteps for MDM, or None.
        """
        z_gru = self.gru(motions)                        # (B, 512)
        z_mdm = self.mdm(motions,
                         timesteps=timesteps,
                         lengths=lengths)                # (B, 512)

        z_combined = torch.cat([z_gru, z_mdm], dim=-1)  # (B, 1024)
        return self.fusion_proj(z_combined)              # (B, text_out_dim)