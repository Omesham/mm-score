# models/timesnet_encoder.py
import torch
import torch.nn as nn

class TimesNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):  # x: (B, T, d)
        x = x.permute(0, 2, 1)  # (B, d, T)
        x = self.conv(x)       # (B, d, T)
        return x.permute(0, 2, 1)  # (B, T, d)


class TimesNetEncoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_blocks=2):
        super().__init__()
        self.blocks = nn.Sequential(*[TimesNetBlock(input_dim, hidden_dim) for _ in range(num_blocks)])

    def forward(self, x):  # x: (B, T, d)
        return self.blocks(x)
