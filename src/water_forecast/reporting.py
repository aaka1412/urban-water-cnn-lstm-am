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
    """MAE/MSE/RMSE/R2 plus MAPE and accuracy_pct = 100 - MAPE (floored at 0).

    Forecasting is regression, so "accuracy" here is the mean percentage closeness to the actual value,
    not a classification hit rate. MAPE is undefined when any actual value is zero.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    r2 = float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan")
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100) if np.all(actual != 0) else float("nan")
    return {"mae": float(mean_absolute_error(actual, predicted)), "mse": float(mean_squared_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5), "r2": r2,
            "mape": mape, "accuracy_pct": max(0.0, 100.0 - mape) if np.isfinite(mape) else float("nan")}


def save_comparison_bars(summary: pd.DataFrame, output: Path, unit: str) -> None:
    """MAE per model: neural models blue, simple baselines grey, overall best green."""
    plt.style.use("seaborn-v0_8-whitegrid")
    ordered = summary.sort_values("mae")
    colors = ["#2ca02c" if i == 0 else ("#7f7f7f" if kind == "baseline" else "#1f77b4")
              for i, kind in enumerate(ordered.get("type", ["neural"] * len(ordered)))]
    fig, axis = plt.subplots(figsize=(10, 4.8))
    bars = axis.bar(ordered["model"], ordered["mae"], color=colors)
    axis.bar_label(bars, labels=[f"{value:,.0f}" if value >= 100 else f"{value:,.2f}" for value in ordered["mae"]],
                   padding=3, fontsize=8.5)
    axis.set(title="Walk-forward MAE (green = best, grey = simple baseline, blue = neural)",
             xlabel="Model", ylabel=f"MAE ({unit})")
    axis.margins(y=0.15)
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=180)
    plt.close(fig)


def save_plots(dates: pd.Series, actual: np.ndarray, predictions: dict[str, np.ndarray], summary: pd.DataFrame,
               output: Path, highlight: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.plot(dates, actual, label="Actual (raw)", color="black", linewidth=1.8)
    for name, values in predictions.items():
        axis.plot(dates, values, label=name.replace("_", " ").title(), alpha=0.8)
    axis.set(title="Daily urban water supply: actual vs walk-forward forecast", ylabel="Supply (m³)")
    axis.legend(ncol=2); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output / "forecast_vs_actual.png", dpi=180); plt.close(fig)
    chosen = predictions[highlight]
    fig, axis = plt.subplots(figsize=(5.5, 5.5)); axis.scatter(actual, chosen, alpha=0.65)
    limits = [min(actual.min(), chosen.min()), max(actual.max(), chosen.max())]; axis.plot(limits, limits, "r--")
    axis.set(xlabel="Actual supply (m³)", ylabel=f"{highlight} forecast (m³)", title="Prediction scatter"); fig.tight_layout(); fig.savefig(output / "scatter.png", dpi=180); plt.close(fig)
    save_comparison_bars(summary, output, "m³")
