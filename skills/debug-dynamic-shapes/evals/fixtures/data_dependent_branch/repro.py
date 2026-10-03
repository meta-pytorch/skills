"""Masked mean pooling over token embeddings, compiled with torch.compile.

Rows with no valid tokens pool to zeros.
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


class MaskedPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x, mask):
        out = []
        for i in range(x.shape[0]):
            idx = torch.nonzero(mask[i]).squeeze(1)
            if idx.shape[0] > 0:
                out.append(x[i, idx].mean(0))
            else:
                out.append(x.new_zeros(x.shape[-1]))
        return self.proj(torch.stack(out))


model = torch.compile(MaskedPool(64), fullgraph=True)

x = torch.randn(4, 16, 64)
mask = torch.rand(4, 16) > 0.5
mask[2] = False
print(model(x, mask).shape)
