import torch
import torch.nn as nn


class LSTMMotionEncoder(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int = 512, out_dim: int = 512, num_layers: int = 2):
        super().__init__()
        # Le LSTM traite la séquence (B, T, D)
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, motions: torch.Tensor) -> torch.Tensor:
        mask = (motions.abs().sum(dim=-1) > 0)
        lengths = mask.sum(dim=1).cpu()

        packed_motions = nn.utils.rnn.pack_padded_sequence(
            motions, lengths, batch_first=True, enforce_sorted=False
        )

        _, (h_n, c_n) = self.lstm(packed_motions)

        last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
