from __future__ import annotations

"""Quick MNIST benchmark for modular-form activations.

Run with

```bash
python -m modular_form_activation.benchmark_mnist --activation eta --epochs 5 --batch-size 128
```

Activations supported:
    • `eta`  – DedekindEtaActivation
    • `relu` – torch.nn.ReLU (baseline)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # progress bars

from .activations import DedekindEtaActivation


@dataclass
class Config:
    activation: Literal["eta", "relu"] = "eta"  # choice of activation
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    truncation: int = 20  # for η activation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: Path = Path.home() / ".cache" / "mnist-data"


class MLP(nn.Module):
    """Simple 3-layer feed-forward network."""

    def __init__(self, activation: nn.Module) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


def get_activation(cfg: Config) -> nn.Module:
    if cfg.activation == "eta":
        return DedekindEtaActivation(truncation=cfg.truncation)
    if cfg.activation == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation '{cfg.activation}'")


def train_one_epoch(model: nn.Module, loader: DataLoader, optimiser: optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, leave=False, desc="train", unit="batch"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimiser.zero_grad()
        output = model(inputs)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        optimiser.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, leave=False, desc="eval", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            pred = output.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    return correct / total


def main(cfg: Config | None = None) -> None:  # noqa: D401
    if cfg is None:
        import argparse

        parser = argparse.ArgumentParser(description="MNIST benchmark for modular-form activation")
        parser.add_argument("--activation", choices=["eta", "relu"], default="eta")
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--truncation", type=int, default=20)
        args = parser.parse_args()
        cfg = Config(
            activation=args.activation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            truncation=args.truncation,
        )

    device = torch.device(cfg.device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MLP(get_activation(cfg)).to(device)
    optimiser = optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Training with activation: {cfg.activation} on {device}\n")
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimiser, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d}: loss = {loss:.4f}, test accuracy = {acc * 100:.2f}%")


if __name__ == "__main__":
    main() 