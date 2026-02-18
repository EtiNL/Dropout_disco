import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUMotionEncoder(nn.Module):
    def __init__(self, d_in, hidden_dim=512, out_dim=512, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, motions):
        # motions: (B, T, D) padded tensor — lengths inferred from zero-padding
        lengths = (motions.abs().sum(dim=-1) > 0).sum(dim=1).clamp_min(1).cpu()

        packed = pack_padded_sequence(motions, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T_real_max, 2H)

        # mean-pool over real frames only
        mask = torch.zeros(output.shape[0], output.shape[1], 1, device=output.device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (output * mask).sum(dim=1) / denom

        return self.fc(pooled)
