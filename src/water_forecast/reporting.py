from __future__ import annotations

import os
from pathlib import Path

# Keep Matplotlib's cache inside the project rather than a protected profile path.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "outputs" / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(actual, predicted)), "mse": float(mean_squared_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5), "r2": float(r2_score(actual, predicted))}


def save_plots(dates: pd.Series, actual: np.ndarray, predictions: dict[str, np.ndarray], summary: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.plot(dates, actual, label="Actual", color="black", linewidth=1.8)
    for name, values in predictions.items(): axis.plot(dates, values, label=name.replace("_", " ").title(), alpha=0.8)
    axis.set(title="Daily urban water supply: actual vs forecast", ylabel="Supply (m³)")
    axis.legend(ncol=2); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output / "forecast_vs_actual.png", dpi=180); plt.close(fig)
    proposed = predictions["cnn_lstm_attention"]
    fig, axis = plt.subplots(figsize=(5.5, 5.5)); axis.scatter(actual, proposed, alpha=0.65)
    limits = [min(actual.min(), proposed.min()), max(actual.max(), proposed.max())]; axis.plot(limits, limits, "r--")
    axis.set(xlabel="Actual supply (m³)", ylabel="CNN-LSTM-Attention forecast (m³)", title="Prediction scatter"); fig.tight_layout(); fig.savefig(output / "scatter.png", dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4.5)); summary.plot.bar(x="model", y="rmse", legend=False, ax=axis, color="#1f77b4")
    axis.set(title="Holdout RMSE by model", xlabel="Model", ylabel="RMSE (m³)"); axis.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(output / "model_comparison.png", dpi=180); plt.close(fig)
