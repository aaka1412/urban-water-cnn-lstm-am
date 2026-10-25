"""Save / load trained forecasters and roll them forward for future days.

The training script only wrote metrics before. This module adds the missing
piece for real use: persist the best model weights + scaler so predict.py and
chatbot.py can answer "what is tomorrow's supply?" in terminal.

Run folders store `model.pt` (weights) and `scaler.json` (min-max parameters).
Folders created before this change used `cnn_lstm_attention.pt` and
`scaler.pkl`; both are still loadable.
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from .models import Forecaster
from .training import ModelParams


def save_bundle(output_dir: Path, model: Forecaster, scaler, params: ModelParams,
                sequence_length: int, target_column: str = "supply_m3") -> Path:
    """Persist model + scaler + metadata inside a run folder. Returns bundle path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "variant": model.variant,
        "params": asdict(params),
        "sequence_length": int(sequence_length),
        "target_column": target_column,
    }
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "scaler.json").write_text(json.dumps(_scaler_to_json(scaler), indent=2))
    (output_dir / "model_bundle.json").write_text(json.dumps(bundle, indent=2))
    return output_dir / "model_bundle.json"


def _scaler_to_json(scaler: MinMaxScaler) -> dict:
    """Min-max parameters as plain JSON so runs stay loadable across Python/sklearn versions."""
    names = [str(name) for name in getattr(scaler, "feature_names_in_", [])]
    return {
        "n_features_in_": int(scaler.n_features_in_),
        "feature_names_in_": names,
        "data_min_": [float(v) for v in np.atleast_1d(scaler.data_min_)],
        "data_max_": [float(v) for v in np.atleast_1d(scaler.data_max_)],
        "data_range_": [float(v) for v in np.atleast_1d(scaler.data_range_)],
        "scale_": [float(v) for v in np.atleast_1d(scaler.scale_)],
        "min_": [float(v) for v in np.atleast_1d(scaler.min_)],
    }


def _scaler_from_json(payload: dict) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.n_features_in_ = int(payload["n_features_in_"])
    if payload.get("feature_names_in_"):
        scaler.feature_names_in_ = np.asarray(payload["feature_names_in_"], dtype=object)
    for key in ("data_min_", "data_max_", "data_range_", "scale_", "min_"):
        setattr(scaler, key, np.asarray(payload[key], dtype=np.float64))
    return scaler


def clean_with_bundle_settings(frame: pd.DataFrame, meta: dict, smoothing_allowed: bool = True) -> pd.DataFrame:
    """Prepare fresh daily data exactly like the run's training inputs (clip limits, causal smoothing)."""
    from .data import clean_daily_supply

    prep = meta.get("preprocessing", {})
    limits = prep.get("clip_limits")
    return clean_daily_supply(
        frame, smoothing=smoothing_allowed and prep.get("smoothing", True),
        window=prep.get("window", 15), polyorder=prep.get("polyorder", 2),
        clip_limits=tuple(limits) if limits else None,
    )


RUN_STAMP = re.compile(r"(\d{8}_\d{6})$")


def run_frequency(run_dir: Path) -> str:
    """"daily" or "two_month_billing_period", read from the run's bundle."""
    meta = json.loads((Path(run_dir) / "model_bundle.json").read_text())
    return meta.get("data_frequency", "daily")


def _run_time(run_dir: Path) -> str:
    # Compare the timestamp in the folder name ("kharghar_20261010_212955" -> "20261010_212955"),
    # not the whole name: "kharghar_..." would otherwise always sort after "2026...".
    match = RUN_STAMP.search(run_dir.name)
    return match.group(1) if match else datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y%m%d_%H%M%S")


def find_latest_run(outputs_dir: Path = Path("outputs"), frequency: str | None = None) -> Path | None:
    """Newest run folder with a model bundle; optionally only runs of one data frequency."""
    if not outputs_dir.exists():
        return None
    candidates = [p for p in outputs_dir.iterdir() if p.is_dir() and (p / "model_bundle.json").exists()]
    if frequency is not None:
        candidates = [p for p in candidates if run_frequency(p) == frequency]
    return max(candidates, key=_run_time) if candidates else None


def weights_file(run_dir: Path) -> Path:
    """model.pt, or cnn_lstm_attention.pt for runs saved before the rename."""
    run_dir = Path(run_dir)
    current = run_dir / "model.pt"
    return current if current.exists() else run_dir / "cnn_lstm_attention.pt"


def load_bundle(run_dir: Path) -> tuple[Forecaster, object, dict]:
    """Load (model, scaler, metadata) from a run folder."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "model_bundle.json").read_text())
    params = ModelParams(**meta["params"])
    model = Forecaster(
        variant=meta.get("variant", "cnn_lstm_attention"),
        hidden_size=params.hidden_size,
        cnn_filters=params.cnn_filters,
        kernel_size=params.kernel_size,
        dropout=params.dropout,
        attention_size=params.attention_size,
    )
    state = torch.load(weights_file(run_dir), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    scaler_json = run_dir / "scaler.json"
    if scaler_json.exists():
        scaler = _scaler_from_json(json.loads(scaler_json.read_text()))
    else:
        with open(run_dir / "scaler.pkl", "rb") as f:  # runs saved before the rename
            scaler = pickle.load(f)
    return model, scaler, meta


def forecast_future(model: Forecaster, scaler, history_m3: pd.Series | np.ndarray,
                    days: int, seq_len: int) -> np.ndarray:
    """Autoregressive forecast: feed each prediction back as input for next day.

    history_m3: at least `seq_len` most recent daily values in m3.
    Returns array of `days` forecasts in m3.
    """
    values = np.asarray(history_m3, dtype=np.float64).ravel()
    if len(values) < seq_len:
        raise ValueError(f"Need at least {seq_len} history values, got {len(values)}.")
    # Use DataFrame with column name to match scaler fitted with feature names.
    scaled = scaler.transform(pd.DataFrame(values.reshape(-1, 1), columns=["supply_m3"])).astype(np.float32).ravel()
    window = scaled[-seq_len:].tolist()
    out_scaled: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(days):
            x = torch.tensor([window[-seq_len:]], dtype=torch.float32).unsqueeze(-1)
            pred = model(x).item()
            out_scaled.append(pred)
            window.append(pred)
    return scaler.inverse_transform(np.asarray(out_scaled).reshape(-1, 1)).ravel()
