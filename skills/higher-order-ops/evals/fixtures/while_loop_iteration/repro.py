"""A model that iterates a fixed-point update until a data-dependent tolerance is met."""

import torch
import torch.nn as nn

torch.manual_seed(0)


class FixedPoint(nn.Module):
    def forward(self, x):
        prev = x + 1
        # Number of iterations depends on the data (iterate to convergence).
        while (x - prev).abs().max() > 1e-4:
            prev = x
            x = torch.cos(x)
        return x


model = FixedPoint().eval()

for scale in (2.0, 0.1):
    x = torch.randn(6) * scale
    eager = model(x.clone())
    compiled = torch.compile(model, fullgraph=True)(x.clone())
    print("max diff:", (eager - compiled).abs().max().item())
