from __future__ import annotations

import optuna
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from .training import ModelParams, train_model


def tune_cnn_lstm_attention(x: np.ndarray, y: np.ndarray, trials: int, epochs: int, seed: int) -> tuple[ModelParams, list[dict]]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splitter = TimeSeriesSplit(n_splits=2)

    def objective(trial: optuna.Trial) -> float:
        params = ModelParams(
            hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 96]),
            cnn_filters=trial.suggest_categorical("cnn_filters", [16, 32, 48]),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7]),
            dropout=trial.suggest_float("dropout", 0.0, 0.35),
            attention_size=trial.suggest_categorical("attention_size", [16, 32, 48]),
            learning_rate=trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
        )
        losses = []
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x)):
            _, loss = train_model("cnn_lstm_attention", x[train_idx], y[train_idx], x[valid_idx], y[valid_idx], params,
                                  epochs=epochs, seed=seed + trial.number * 10 + fold, patience=max(4, epochs // 3))
            losses.append(loss)
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    best = ModelParams(**study.best_params)
    history = [{"trial": t.number, "value": t.value, **t.params} for t in study.trials if t.value is not None]
    return best, history
