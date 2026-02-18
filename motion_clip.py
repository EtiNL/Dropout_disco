# motion_clip.py
import numpy as np

import torch
import torch.nn as nn


class MotionClip(nn.Module):
    def __init__(self, motion_model, text_dim, text_model_name=None):
        super().__init__()
        self.motion_model = motion_model
        self.text_dim = text_dim
        self.text_model_name = text_model_name  # optional bookkeeping
        self.proj = None
        self._in_dim = None
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, motions, **motion_kwargs):
        # Forward any kwargs (e.g., lengths=...) to the motion encoder
        z = self.motion_model(motions, **motion_kwargs)  # (B,M)
        if z.ndim != 2:
            raise ValueError(f"motion_model must return (B, M). Got {z.shape}.")
        if self._in_dim is None:
            self._in_dim = z.shape[1]
            self.proj = nn.Identity() if self._in_dim == self.text_dim else nn.Linear(self._in_dim, self.text_dim)
            self.proj = self.proj.to(z.device)
        return self.proj(z)
