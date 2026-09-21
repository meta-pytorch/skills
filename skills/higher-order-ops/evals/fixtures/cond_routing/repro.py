"""A model that routes to one of two experts based on a data-dependent gate."""

import torch
import torch.nn as nn

torch.manual_seed(0)


class Router(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expert_a = nn.Linear(dim, dim)
        self.expert_b = nn.Linear(dim, dim)

    def forward(self, x):
        # Only the selected expert should run for a given input.
        if x.mean() > 0:
            return self.expert_a(x).relu()
        return self.expert_b(x).tanh()


model = Router(8).eval()

for x in (torch.randn(4, 8) + 3, torch.randn(4, 8) - 3):
    eager = model(x)
    compiled = torch.compile(model, fullgraph=True)(x)
    print("max diff:", (eager - compiled).abs().max().item())
