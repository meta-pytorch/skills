"""Inference model with a hand-written, numerically-stable log-softmax head (fp16)."""

import torch
import torch.nn as nn


def stable_log_softmax(x):
    m = x.max(dim=-1, keepdim=True).values
    z = x - m
    return z - z.exp().sum(dim=-1, keepdim=True).log()


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return stable_log_softmax(self.proj(x))


torch.manual_seed(0)
model = Model(512).half().eval()
x = torch.randn(64, 512).half() * 32

with torch.no_grad():
    eager = model(x)
    compiled = torch.compile(model, fullgraph=True)(x)
print("max diff:", (eager - compiled).abs().max().item())
