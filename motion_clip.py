import numpy as np

import torch
import torch.nn as nn


class MotionClip(nn.Module):
    """Wraps a motion encoder and projects it into the text embedding space."""

    def __init__(self, motion_model, text_dim: int, text_model_name: str | None = None):
        super().__init__()
        self.motion_model = motion_model
        self.text_dim = int(text_dim)
        self.text_model_name = text_model_name  # optional bookkeeping

        self.proj = None
        self._in_dim = None

        # CLIP-style temperature (log scale)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, motions, **motion_kwargs):
        """
        motions:
          - padded tensor (B,T,D)
          - or list of (T_i,D) tensors if your motion_model supports it

        motion_kwargs: forwarded to motion_model (e.g., lengths=...)
        """
        z = self.motion_model(motions, **motion_kwargs)  # (B,M)
        if z.ndim != 2:
            raise ValueError(f"motion_model must return (B, M). Got {z.shape}.")

        if self._in_dim is None:
            self._in_dim = int(z.shape[1])
            if self._in_dim == self.text_dim:
                self.proj = nn.Identity()
            else:
                self.proj = nn.Linear(self._in_dim, self.text_dim)
            self.proj = self.proj.to(z.device)

        return self.proj(z)
