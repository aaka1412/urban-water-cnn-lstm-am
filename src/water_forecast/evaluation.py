"""Leak-free evaluation: walk-forward (rolling-origin) testing, equal tuning, and naive baselines.

Rules this module enforces:
  * A fold's test targets are never used for training, early stopping, tuning, scaling or clipping.
    Each fold refits its preprocessing and scaler on data before its train_end.
  * Every neural variant gets the same tuning budget, on data that ends before the first test point.
  * Baselines are scored on exactly the same test points as the neural models.
  * Forecasts are compared with the raw observed values, never with smoothed ones.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .baselines import naive_forecasts
from .reporting import metrics
from .training import ModelParams, fit_with_validation, predict
from .tuning import tune_variant

VARIANTS = ("lstm", "cnn_lstm", "lstm_attention", "cnn_lstm_attention")

# (raw_values, train_end) -> (model_inputs, training_targets). Any fitted statistic may use only
# raw_values[:train_end], and model_inputs[t] may depend only on raw_values[:t + 1].
Preprocess = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]


def identity_preprocess(raw: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=float)
    return values, values


@dataclass(frozen=True)
class Fold:
    train_end: int  # positions < train_end are the only data the fold may learn from
    test_indices: tuple[int, ...]  # forecast origins; each is predicted one step ahead (or rolled forward)


def _scaler(values: np.ndarray) -> MinMaxScaler:
    return MinMaxScaler().fit(pd.DataFrame({"supply_m3": values}))


def _scale(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.transform(pd.DataFrame({"supply_m3": values})).astype(np.float32).ravel()


def _unscale(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(values, dtype=np.float64).reshape(-1, 1)).ravel()


def _windows(inputs: np.ndarray, targets: np.ndarray, seq_len: int, ends: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([inputs[end - seq_len:end] for end in ends], dtype=np.float32)[..., None]
    y = np.asarray([targets[end] for end in ends], dtype=np.float32)
    return x, y


def fit_on_prefix(variant: str, params: ModelParams, raw: np.ndarray, train_end: int, seq_len: int,
                  preprocess: Preprocess, epochs: int, seed: int):
    """Train on positions before train_end only. Returns (model, scaler, scaled_inputs, best_epoch)."""
    inputs, targets = preprocess(raw, train_end)
    scaler = _scaler(inputs[:train_end])
    scaled_inputs, scaled_targets = _scale(scaler, inputs), _scale(scaler, targets)
    x, y = _windows(scaled_inputs, scaled_targets, seq_len, range(seq_len, train_end))
    model, best_epoch = fit_with_validation(variant, x, y, params, epochs, seed)
    return model, scaler, scaled_inputs, best_epoch


def _roll_forward(model, scaled_inputs: np.ndarray, origin: int, seq_len: int, horizon: int) -> np.ndarray:
    """Forecast origin .. origin+horizon-1 from observed inputs before origin, feeding predictions back."""
    window = list(scaled_inputs[origin - seq_len:origin])
    out = []
    for _ in range(horizon):
        step = float(predict(model, np.asarray(window[-seq_len:], dtype=np.float32)[None, :, None])[0])
        out.append(step)
        window.append(step)
    return np.asarray(out)


def tune_all(raw: np.ndarray, train_end: int, seq_len: int, preprocess: Preprocess, trials: int,
             epochs: int, seed: int, variants: Sequence[str] = VARIANTS) -> tuple[dict[str, ModelParams], list[dict]]:
    """Tune every variant with the same budget on data before train_end (the first test point)."""
    inputs, targets = preprocess(raw, train_end)
    scaler = _scaler(inputs[:train_end])
    x, y = _windows(_scale(scaler, inputs), _scale(scaler, targets), seq_len, range(seq_len, train_end))
    tuned, history = {}, []
    for variant in variants:
        tuned[variant], rows = tune_variant(variant, x, y, trials, epochs, seed)
        history.extend(rows)
    return tuned, history


def walk_forward(raw: np.ndarray, folds: Sequence[Fold], seq_len: int, tuned: dict[str, ModelParams],
                 preprocess: Preprocess, epochs: int, runs: int, seed: int, horizon: int, ma_window: int,
                 season: int, progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Score every neural variant (runs seeds each) and every baseline on the same walk-forward folds.

    Returns one row per (model, run, origin, horizon) with the forecast and the raw actual value.
    """
    raw = np.asarray(raw, dtype=float)
    rows = []
    for number, fold in enumerate(folds, 1):
        if progress:
            progress(f"Fold {number}/{len(folds)}: training on {fold.train_end} points, "
                     f"testing {len(fold.test_indices)} point(s)")
        targets_for = {origin: [origin + step for step in range(horizon) if origin + step < len(raw)]
                       for origin in fold.test_indices}
        for origin, target_positions in targets_for.items():
            for name, values in naive_forecasts(raw, origin, len(target_positions), ma_window, season).items():
                for step, (position, value) in enumerate(zip(target_positions, values), 1):
                    if not np.isnan(value):
                        rows.append({"model": name, "type": "baseline", "run": 0, "origin": origin, "horizon": step,
                                     "target_index": position, "actual": raw[position], "forecast": float(value)})
        for variant, params in tuned.items():
            for run in range(runs):
                model, scaler, scaled_inputs, best_epoch = fit_on_prefix(
                    variant, params, raw, fold.train_end, seq_len, preprocess, epochs, seed + run)
                for origin, target_positions in targets_for.items():
                    forecast = _unscale(scaler, _roll_forward(model, scaled_inputs, origin, seq_len,
                                                              len(target_positions)))
                    for step, (position, value) in enumerate(zip(target_positions, forecast), 1):
                        rows.append({"model": variant, "type": "neural", "run": run, "origin": origin,
                                     "horizon": step, "target_index": position, "actual": raw[position],
                                     "forecast": float(value), "best_epoch": best_epoch})
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame, mase_scale: float, horizon: int = 1) -> pd.DataFrame:
    """Metrics per model at one horizon, averaged over seeds; MASE = MAE / in-sample naive MAE."""
    subset = predictions[predictions["horizon"] == horizon]
    rows = []
    for (model, kind), group in subset.groupby(["model", "type"]):
        per_run = [metrics(run["actual"].to_numpy(), run["forecast"].to_numpy()) for _, run in group.groupby("run")]
        maes = [record["mae"] for record in per_run]
        rows.append({
            "model": model, "type": kind, "runs": len(per_run), "n_forecasts": len(group) // len(per_run),
            **{name: float(np.mean([record[name] for record in per_run])) for name in per_run[0]},
            "mase": float(np.mean(maes) / mase_scale) if mase_scale > 0 else float("nan"),
            "mae_run_std": float(np.std(maes)),
        })
    report = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    # MASE's in-sample scale can be inflated by one past spike; relMAE compares against repeating the
    # last value on exactly the same test points (< 1 = better than that naive forecast).
    naive = report.loc[report["model"] == "naive_last_value", "mae"]
    report["rel_mae"] = report["mae"] / naive.iloc[0] if not naive.empty else float("nan")
    return report


def horizon_table(predictions: pd.DataFrame, mase_scale: float) -> pd.DataFrame:
    """MAE and MASE for each model at each forecast horizon (1 = next step, 2 = two steps ahead, ...)."""
    rows = []
    for (model, kind, step), group in predictions.groupby(["model", "type", "horizon"]):
        mae = float(np.mean([np.mean(np.abs(run["actual"] - run["forecast"])) for _, run in group.groupby("run")]))
        rows.append({"model": model, "type": kind, "horizon": int(step), "n_forecasts": group["origin"].nunique(),
                     "mae": mae, "mase": mae / mase_scale if mase_scale > 0 else float("nan")})
    return pd.DataFrame(rows).sort_values(["horizon", "mae"]).reset_index(drop=True)


def naive_scale(raw: np.ndarray, train_end: int) -> float:
    """Mean absolute one-step change before the first test point: the denominator of MASE."""
    return float(np.mean(np.abs(np.diff(np.asarray(raw[:train_end], dtype=float)))))


def mean_forecasts(predictions: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Seed-averaged forecast per model and origin, wide format, for plots and the predictions CSV."""
    subset = predictions[predictions["horizon"] == horizon]
    wide = subset.pivot_table(index="target_index", columns="model", values="forecast", aggfunc="mean")
    wide.insert(0, "actual", subset.groupby("target_index")["actual"].first())
    return wide.reset_index()


def verdict(report: pd.DataFrame) -> str:
    """One-paragraph plain reading: does the best neural model beat the best simple baseline?"""
    neural, baseline = report[report["type"] == "neural"], report[report["type"] == "baseline"]
    if neural.empty or baseline.empty:
        return ""
    best_nn, best_base = neural.iloc[0], baseline.iloc[0]
    nn_mae, base_mae = best_nn["mae"], best_base["mae"]
    lines = [f"Best neural model: {best_nn['model']} (MAE {nn_mae:,.2f}, relMAE {best_nn['rel_mae']:.3f}).",
             f"Best simple baseline: {best_base['model']} (MAE {base_mae:,.2f}, relMAE {best_base['rel_mae']:.3f})."]
    if abs(base_mae - nn_mae) <= max(best_nn["mae_run_std"], 0.02 * base_mae):
        lines.append(f"Verdict: no clear difference ({100 * (nn_mae - base_mae) / base_mae:+.1f}% MAE); the gap is "
                     "within seed-to-seed variation or under 2%, so the neural model does not clearly beat the baseline.")
    elif nn_mae < base_mae:
        lines.append(f"Verdict: the neural model has {100 * (base_mae - nn_mae) / base_mae:.1f}% lower error "
                     "than the best baseline.")
    else:
        lines.append(f"Verdict: the simple baseline has {100 * (nn_mae - base_mae) / nn_mae:.1f}% lower error "
                     "than the best neural model.")
    return "\n".join(lines)
