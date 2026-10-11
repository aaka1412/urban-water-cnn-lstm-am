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


def train_model(variant: str, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray | None,
                y_valid: np.ndarray | None, params: ModelParams, epochs: int, seed: int,
                patience: int = 15) -> tuple[Forecaster, float, int]:
    """Train one forecaster; returns (model, best validation loss, epoch count of the kept weights).

    With a validation set, training stops early and keeps the best-validation weights. The validation
    windows must never be test windows. Pass x_valid=None to train for exactly `epochs` epochs.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Forecaster(variant=variant, **{k: v for k, v in asdict(params).items() if k not in {"learning_rate", "batch_size"}}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=params.batch_size, shuffle=True)
    best_state, best_loss, best_epoch, stalled = None, float("inf"), epochs, 0
    validate = x_valid is not None and y_valid is not None
    if validate:
        valid_x, valid_y = torch.from_numpy(x_valid).to(device), torch.from_numpy(y_valid).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if not validate:
            continue
        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(valid_x), valid_y).item()
        if valid_loss < best_loss - 1e-7:
            best_loss, best_epoch, stalled = valid_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stalled += 1
            if stalled >= patience:
                break
    if not validate:
        return model, float("nan"), epochs
    model.load_state_dict(best_state)
    return model, best_loss, best_epoch


def fit_with_validation(variant: str, x: np.ndarray, y: np.ndarray, params: ModelParams, epochs: int, seed: int,
                        val_fraction: float = 0.2, min_val: int = 2) -> tuple[Forecaster, int]:
    """Pick the epoch count on the most recent training windows, then refit on all of them.

    x/y must contain only pre-test windows in time order. The last `val_fraction` of them act as the
    early-stopping validation set, so the test period never influences which weights are kept.
    """
    n_val = max(min_val, int(round(len(x) * val_fraction)))
    if len(x) - n_val < 2:
        raise ValueError(f"Need more training windows for early stopping: got {len(x)}, validation uses {n_val}.")
    _, _, best_epoch = train_model(variant, x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:], params, epochs, seed)
    model, _, _ = train_model(variant, x, y, None, None, params, best_epoch, seed)
    return model, best_epoch


def predict(model: Forecaster, x: np.ndarray) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.from_numpy(x).to(device)).cpu().numpy()
