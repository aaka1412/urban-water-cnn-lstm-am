from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import Forecaster


@dataclass(frozen=True)
class ModelParams:
    hidden_size: int = 64
    cnn_filters: int = 32
    kernel_size: int = 3
    dropout: float = 0.15
    attention_size: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_model(variant: str, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray,
                params: ModelParams, epochs: int, seed: int, patience: int = 15) -> tuple[Forecaster, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Forecaster(variant=variant, **{k: v for k, v in asdict(params).items() if k not in {"learning_rate", "batch_size"}}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=params.batch_size, shuffle=True)
    best_state, best_loss, stalled = None, float("inf"), 0
    valid_x, valid_y = torch.from_numpy(x_valid).to(device), torch.from_numpy(y_valid).to(device)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(valid_x), valid_y).item()
        if valid_loss < best_loss - 1e-7:
            best_loss, stalled = valid_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stalled += 1
            if stalled >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_loss


def predict(model: Forecaster, x: np.ndarray) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.from_numpy(x).to(device)).cpu().numpy()
