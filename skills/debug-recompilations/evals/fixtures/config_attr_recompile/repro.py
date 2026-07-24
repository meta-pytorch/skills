"""A small transformer-style stack compiled with torch.compile."""

import torch
import torch.nn as nn

torch.manual_seed(0)


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        h = self.qkv(x)
        head_dim = d // self.n_heads
        h = h.view(b, t, self.n_heads, head_dim)
        h = h * (head_dim ** -0.5)
        return h.reshape(b, t, d)


class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.attn(x))


blocks = [Block(512, 4), Block(512, 8), Block(512, 16)]


@torch.compile
def run(block, x):
    return block(x)


x = torch.randn(2, 16, 512)
for blk in blocks:
    run(blk, x)
