"""PyTorch model, dataset and training loop."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class EngineFaultNet(nn.Module):
    """Feed-forward classifier: input -> 64 -> 32 -> 1 (logit)."""

    def __init__(self, n_features: int, hidden: tuple[int, int] = (64, 32), dropout: float = 0.2):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


@dataclass
class TrainResult:
    model: EngineFaultNet
    best_epoch: int
    epochs_run: int
    history: list[dict[str, float]]


def train_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 200,
    patience: int = 10,
    dropout: float = 0.2,
    use_pos_weight: bool = True,
    seed: int = 42,
) -> TrainResult:
    """Train with Adam and early stopping on validation loss; returns the best weights."""
    seed_everything(seed)
    model = EngineFaultNet(X_train.shape[1], dropout=dropout)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor(neg / pos if use_pos_weight and pos > 0 else 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        SensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, generator=generator
    )
    Xv = torch.as_tensor(X_val, dtype=torch.float32)
    yv = torch.as_tensor(y_val, dtype=torch.float32)

    best_loss, best_state, best_epoch, wait = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    history: list[dict[str, float]] = []
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(yb)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        history.append({"epoch": epoch, "train_loss": total / len(y_train), "val_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model=model, best_epoch=best_epoch, epochs_run=epoch, history=history)


def predict_proba(model: EngineFaultNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()
