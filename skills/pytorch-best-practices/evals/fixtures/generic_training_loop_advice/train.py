import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(batch_size=16):
    inputs = torch.randn(512, 32)
    labels = (inputs[:, 0] + inputs[:, 1] > 0).long()
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train():
    device = torch.device("cpu")
    model = TinyClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader()

    model.train()
    for epoch in range(3):
        total_loss = 0.0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch={epoch} loss={total_loss / len(loader):.4f}")


if __name__ == "__main__":
    train()
