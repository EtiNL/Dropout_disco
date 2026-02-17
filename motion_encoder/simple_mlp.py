import torch
import torch.nn as nn


class SimpleMotionEncoder(nn.Module):
    """
    Minimal baseline motion encoder:
      (B, T, D) -> (B, M)

    Steps:
      1) frame MLP: x_t -> h_t
      2) masked mean pooling over time (ignores zero-padded frames)
      3) output embedding (B, out_dim)

    Assumption for masking:
      padded frames are exactly all-zeros in the input (as in our padding code).
    """
    def __init__(self, d_in: int, hidden: int = 256, out_dim: int = 512):
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, motions: torch.Tensor) -> torch.Tensor:
        # motions: (B, T, D)
        if motions.ndim != 3:
            raise ValueError(f"Expected motions shape (B,T,D), got {motions.shape}")

        x = motions
        h = self.frame_mlp(x)  # (B, T, out_dim)

        # mask padded frames: True where frame is non-zero
        mask = (x.abs().sum(dim=-1) > 0).float()  # (B, T)

        # masked mean pooling
        h = h * mask.unsqueeze(-1)                # (B, T, out_dim)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
        z = h.sum(dim=1) / denom                  # (B, out_dim)

        return z
