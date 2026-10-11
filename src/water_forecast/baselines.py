"""Simple reference forecasts. A neural model is only worth using if it beats these."""
from __future__ import annotations

import numpy as np


def naive_forecasts(history: np.ndarray, origin: int, horizon: int, ma_window: int,
                    season: int) -> dict[str, np.ndarray]:
    """Forecasts for positions origin .. origin+horizon-1 using only history[:origin].

    naive_last_value      repeat the last observed value
    moving_average_<k>    average of the last k observed values
    seasonal_naive_<m>    value from one season earlier (e.g. same bill last year); only if available
    """
    past = np.asarray(history[:origin], dtype=float)
    forecasts = {
        "naive_last_value": np.full(horizon, past[-1]),
        f"moving_average_{ma_window}": np.full(horizon, past[-ma_window:].mean()),
    }
    if origin >= season:
        # For steps beyond one season ahead the "same time last season" value is itself unknown.
        forecasts[f"seasonal_naive_{season}"] = np.array(
            [past[origin + step - season] if step < season else np.nan for step in range(horizon)]
        )
    return forecasts
