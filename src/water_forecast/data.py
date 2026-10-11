from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from scipy.signal import savgol_coeffs


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


def regularize_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day; missing days take the last known reading (never a later one)."""
    clean = frame.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["supply_m3"] = pd.to_numeric(clean["supply_m3"], errors="coerce")
    clean = clean.dropna().groupby("date", as_index=False)["supply_m3"].mean().sort_values("date")
    if clean.empty:
        raise ValueError("No valid date/supply observations remain after cleaning")
    daily = clean.set_index("date").asfreq("D")
    daily["supply_m3"] = daily["supply_m3"].ffill()
    return daily.reset_index()


def train_clip_limits(values: np.ndarray, train_end: int, lower: float = 0.01,
                      upper: float = 0.99) -> tuple[float, float]:
    """Outlier limits from the training part only (values[:train_end]), so test days cannot shape them."""
    train = np.asarray(values[:train_end], dtype=float)
    return float(np.quantile(train, lower)), float(np.quantile(train, upper))


def causal_savgol(values: np.ndarray, window: int = 15, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing that looks only backwards.

    Each day's value comes from a polynomial fitted to that day and the previous window-1 days, evaluated
    at the newest day. A centred filter would also use the next days, i.e. information from the future.
    The first window-1 days are left unsmoothed.
    """
    values = np.asarray(values, dtype=float)
    window = window if window % 2 else window - 1
    out = values.copy()
    if window <= polyorder or len(values) < window:
        return out
    coeffs = savgol_coeffs(window, polyorder, pos=window - 1, use="dot")
    out[window - 1:] = sliding_window_view(values, window) @ coeffs
    return out


def clean_daily_supply(frame: pd.DataFrame, smoothing: bool, window: int = 15, polyorder: int = 2,
                       clip_limits: tuple[float, float] | None = None) -> pd.DataFrame:
    """Regularize a daily series, optionally clip with given limits, optionally smooth causally.

    Every step uses only the current and earlier days. Fit clip_limits on training data only
    (see train_clip_limits); without them no clipping is applied.
    """
    daily = regularize_daily(frame)
    values = daily["supply_m3"].to_numpy(dtype=float)
    if clip_limits is not None:
        values = np.clip(values, *clip_limits)
    if smoothing:
        values = causal_savgol(values, window, polyorder)
    daily["supply_m3"] = values
    return daily
