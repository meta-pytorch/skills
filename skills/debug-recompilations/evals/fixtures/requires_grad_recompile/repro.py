"""Training an MLP with a periodic validation pass, compiled with torch.compile.

The training step also records a saliency map from the input gradients.
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


model = MLP(16)
run = torch.compile(model)

train_batch = torch.randn(8, 16, requires_grad=True)
run(train_batch).sum().backward()
saliency = train_batch.grad.abs().mean(0)

val_batch = torch.randn(8, 16, requires_grad=False)
run(val_batch)
