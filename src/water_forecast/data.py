from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_dates: pd.Series
    scaler: MinMaxScaler


def make_synthetic_supply(days: int = 1_460, seed: int = 42) -> pd.DataFrame:
    """Create a documented fallback dataset with annual/weekly patterns and event peaks."""
    if days < 200:
        raise ValueError("days must be at least 200 to support temporal splitting")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    t = np.arange(days)
    annual = 110 * np.sin(2 * np.pi * t / 365.25 - 0.6)
    weekly = 28 * np.sin(2 * np.pi * t / 7)
    trend = 0.045 * t
    heatwave = np.where((t % 365 > 120) & (t % 365 < 155), 65, 0)
    random_events = rng.binomial(1, 0.035, days) * rng.normal(90, 22, days)
    noise = rng.normal(0, 24, days)
    supply = np.maximum(300, 1_120 + annual + weekly + trend + heatwave + random_events + noise)
    return pd.DataFrame({"date": dates, "supply_m3": supply.round(2)})


def load_supply_csv(path: Path, date_column: str, target_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {date_column, target_column} - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return frame[[date_column, target_column]].rename(columns={date_column: "date", target_column: "supply_m3"})


def clean_daily_supply(frame: pd.DataFrame, smoothing: bool, window: int = 15, polyorder: int = 2) -> pd.DataFrame:
    """Clean and regularize a one-value-per-day supply series."""
    clean = frame.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["supply_m3"] = pd.to_numeric(clean["supply_m3"], errors="coerce")
    clean = clean.dropna().groupby("date", as_index=False)["supply_m3"].mean().sort_values("date")
    if clean.empty:
        raise ValueError("No valid date/supply observations remain after cleaning")
    daily = clean.set_index("date").asfreq("D")
    daily["supply_m3"] = daily["supply_m3"].interpolate(limit_direction="both")
    q1, q3 = daily["supply_m3"].quantile([0.01, 0.99])
    daily["supply_m3"] = daily["supply_m3"].clip(q1, q3)
    if smoothing:
        effective_window = min(window if window % 2 else window - 1, len(daily) if len(daily) % 2 else len(daily) - 1)
        if effective_window > polyorder:
            daily["supply_m3"] = savgol_filter(daily["supply_m3"].to_numpy(), effective_window, polyorder)
    return daily.reset_index()


def prepare_windows(frame: pd.DataFrame, sequence_length: int, test_fraction: float = 0.2) -> PreparedData:
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    if len(frame) < sequence_length * 4:
        raise ValueError("Not enough daily records for the selected sequence length")
    split_index = int(len(frame) * (1 - test_fraction))
    scaler = MinMaxScaler()
    scaler.fit(frame.loc[: split_index - 1, ["supply_m3"]])
    values = scaler.transform(frame[["supply_m3"]]).astype(np.float32).ravel()
    x, y, target_indices = [], [], []
    for end in range(sequence_length, len(values)):
        x.append(values[end - sequence_length : end])
        y.append(values[end])
        target_indices.append(end)
    x_array = np.asarray(x, dtype=np.float32)[..., None]
    y_array = np.asarray(y, dtype=np.float32)
    test_mask = np.asarray(target_indices) >= split_index
    return PreparedData(
        x_train=x_array[~test_mask], y_train=y_array[~test_mask],
        x_test=x_array[test_mask], y_test=y_array[test_mask],
        test_dates=frame.loc[np.asarray(target_indices)[test_mask], "date"].reset_index(drop=True), scaler=scaler,
    )
