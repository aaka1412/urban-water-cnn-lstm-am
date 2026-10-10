"""Save / load trained forecasters and roll them forward for future days.

The training script only wrote metrics before. This module adds the missing
piece for real use: persist the best CNN-LSTM-Attention weights + scaler so
predict.py and chatbot.py can answer "what is tomorrow's supply?" in terminal.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    torch.save(model.state_dict(), output_dir / "cnn_lstm_attention.pt")
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (output_dir / "model_bundle.json").write_text(json.dumps(bundle, indent=2))
    return output_dir / "model_bundle.json"


def find_latest_run(outputs_dir: Path = Path("outputs")) -> Path | None:
    """Return newest timestamped run folder that contains a model bundle."""
    if not outputs_dir.exists():
        return None
    candidates = [p for p in outputs_dir.iterdir()
                  if p.is_dir() and (p / "model_bundle.json").exists()]
    if not candidates:
        # Fall back to newest run even without bundle (old runs before this patch).
        dated = [p for p in outputs_dir.iterdir()
                 if p.is_dir() and not p.name.startswith(".")]
        return max(dated) if dated else None
    return max(candidates)


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
    state = torch.load(run_dir / "cnn_lstm_attention.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with open(run_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler, meta


def forecast_future(model: Forecaster, scaler, history_m3: pd.Series | np.ndarray,
                    days: int = 7) -> np.ndarray:
    """Autoregressive forecast: feed each prediction back as input for next day.

    history_m3: at least `sequence_length` most recent daily values in m3.
    Returns array of `days` forecasts in m3.
    """
    values = np.asarray(history_m3, dtype=np.float64).ravel()
    # Use DataFrame with column name to match scaler fitted with feature names.
    scaled = scaler.transform(pd.DataFrame(values.reshape(-1, 1), columns=["supply_m3"])).astype(np.float32).ravel()
    # Sequence length = model-agnostic: caller passes full history, we use
    # the last `seq_len` where seq_len is detected from saved meta by wrapper.
    # Here we require the caller to trim; default to last 30 if unknown.
    seq_len = getattr(forecast_future, "_seq_len", 30)
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


def forecast_with_seq_len(model, scaler, history_m3, seq_len: int, days: int) -> np.ndarray:
    forecast_future._seq_len = int(seq_len)
    try:
        return forecast_future(model, scaler, history_m3, days=days)
    finally:
        forecast_future._seq_len = 30
