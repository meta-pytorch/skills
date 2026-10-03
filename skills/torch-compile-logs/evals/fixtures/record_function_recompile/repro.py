"""An instrumented model compiled with torch.compile."""

import torch
import torch.nn as nn

torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.step = 0

    def forward(self, x):
        with torch.profiler.record_function(f"model_forward_step_{self.step}"):
            return torch.relu(self.lin(x))


model = Model(16)
run = torch.compile(model)

x = torch.randn(4, 16)
for i in range(5):
    model.step = i
    run(x)
