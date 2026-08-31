"""A model that calls a SciPy special function, compiled with fullgraph=True."""

import torch
import torch.nn as nn
import scipy.special as sp


class Model(nn.Module):
    def forward(self, x):
        # complete elliptic integral of the first kind
        y = sp.ellipk(x.cpu().numpy())
        return torch.from_numpy(y).to(x.device, x.dtype)


model = Model()
x = torch.rand(16) * 0.9

eager = model(x)
compiled = torch.compile(model, fullgraph=True)(x)
print("max diff:", (eager - compiled).abs().max().item())
