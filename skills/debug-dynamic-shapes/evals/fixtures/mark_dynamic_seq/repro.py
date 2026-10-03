"""A transformer encoder block run over a batch of sequences, compiled with torch.compile."""

import torch
import torch.nn as nn

torch.manual_seed(0)


class Encoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.fc2(torch.relu(self.fc1(x)))
        return self.norm(x + h)


model = torch.compile(Encoder(64))

for seq_len in [16, 32, 48, 64, 24]:
    x = torch.randn(4, seq_len, 64)
    model(x)
