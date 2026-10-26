from __future__ import annotations

import optuna
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from .training import ModelParams, train_model


def tune_variant(variant: str, x: np.ndarray, y: np.ndarray, trials: int, epochs: int,
                 seed: int, n_splits: int = 5) -> tuple[ModelParams, list[dict]]:
    """Bayesian (TPE) search for one model variant, scored by chronological (time-series) k-fold CV.

    x/y must end before the test period. Every variant gets the same trial budget and only searches
    the settings it actually uses (no CNN settings for LSTM, no attention settings without attention),
    so model comparisons reflect architecture rather than unequal tuning effort.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splitter = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        values = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 96]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.35),
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
        }
        if "cnn" in variant:
            values["cnn_filters"] = trial.suggest_categorical("cnn_filters", [16, 32, 48])
            values["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
        if "attention" in variant:
            values["attention_size"] = trial.suggest_categorical("attention_size", [16, 32, 48])
        params = ModelParams(**values)
        losses = []
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x)):
            _, loss, _ = train_model(variant, x[train_idx], y[train_idx], x[valid_idx], y[valid_idx], params,
                                     epochs=epochs, seed=seed + trial.number * 10 + fold, patience=max(4, epochs // 3))
            losses.append(loss)
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    best = ModelParams(**study.best_params)
    history = [{"variant": variant, "trial": t.number, "value": t.value, **t.params}
               for t in study.trials if t.value is not None]
    return best, history
