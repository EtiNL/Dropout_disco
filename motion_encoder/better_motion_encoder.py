import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    """Masked attention pooling over time."""
    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h:    (B,T,D)
        mask: (B,T) float/bool, 1 for valid, 0 for pad
        """
        if mask.dtype != torch.bool:
            m = mask > 0
        else:
            m = mask

        logits = self.score(h).squeeze(-1)                 # (B,T)
        logits = logits.masked_fill(~m, float("-inf"))     # mask pads
        w = torch.softmax(logits, dim=1)                   # (B,T)
        w = torch.nan_to_num(w, nan=0.0)                   # if all masked (shouldn't happen)
        return torch.einsum("bt,btd->bd", w, h)            # (B,D)


class ResidualMLP(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BetterMotionEncoder(nn.Module):
    """
    More appropriate than per-frame MLP + masked mean:
      - pre-norm
      - 2-layer per-frame MLP with dropout
      - temporal modeling via lightweight 1D conv (depthwise)
      - masked attention pooling (learned aggregation)
      - final projection + L2 normalization (optional)
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        out_dim: int = 512,
        n_res_blocks: int = 2,
        dropout: float = 0.1,
        use_l2norm: bool = False,
    ):
        super().__init__()
        self.use_l2norm = use_l2norm

        self.in_norm = nn.LayerNorm(d_in)

        self.frame_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Lightweight temporal mixing: depthwise conv over time (per channel)
        self.temporal = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

        self.blocks = nn.Sequential(*[ResidualMLP(d_model, hidden=4 * d_model, dropout=dropout)
                                      for _ in range(n_res_blocks)])

        self.pool = AttnPool(d_model, hidden=d_model, dropout=dropout)

        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, motions: torch.Tensor) -> torch.Tensor:
        """
        motions: (B,T,D) padded with zeros
        returns: (B,out_dim)
        """
        # mask padded frames
        mask = (motions.abs().sum(dim=-1) > 0)  # (B,T) bool

        x = self.in_norm(motions)               # (B,T,D)
        h = self.frame_proj(x)                  # (B,T,d_model)

        # temporal conv expects (B,C,T)
        ht = h.transpose(1, 2)                  # (B,d_model,T)
        ht = ht + self.temporal(ht)             # residual temporal mixing
        h = ht.transpose(1, 2)                  # (B,T,d_model)

        h = self.blocks(h)                      # (B,T,d_model)

        z = self.pool(h, mask)                  # (B,d_model)
        z = self.out(z)                         # (B,out_dim)

        if self.use_l2norm:
            z = F.normalize(z, dim=-1)

        return z
