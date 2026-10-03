"""A model that aligns sequence length to a multiple of 8, compiled with torch.compile.

The sequence dimension is marked dynamic to avoid recompiling on every new length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)

    def forward(self, x):
        b, s, h = x.shape
        if s % 8 != 0:
            pad = 8 - s % 8
            x = F.pad(x, (0, 0, 0, pad))
        return self.fc(torch.relu(x))


model = torch.compile(Model(64))

for seq_len in [16, 20, 30, 44, 50]:
    x = torch.randn(4, seq_len, 64)
    torch._dynamo.mark_dynamic(x, 1)
    model(x)
