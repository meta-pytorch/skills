"""A transformer encoder block run over a batch of sequences, compiled with torch.compile.

All input dimensions are marked dynamic to avoid recompiling when shapes change.
"""

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

for batch, seq_len in [(4, 16), (8, 32), (2, 48)]:
    x = torch.randn(batch, seq_len, 64)
    for dim in range(x.dim()):
        torch._dynamo.mark_dynamic(x, dim)
    model(x)
