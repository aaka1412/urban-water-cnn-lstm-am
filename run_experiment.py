from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from water_forecast.data import clean_daily_supply, load_supply_csv, make_synthetic_supply, prepare_windows
from water_forecast.inference import save_bundle
from water_forecast.reporting import metrics, save_plots
from water_forecast.training import ModelParams, predict, train_model
from water_forecast.tuning import tune_cnn_lstm_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate CNN-LSTM-Attention water-supply forecasters.")
    parser.add_argument("--data", type=Path, help="Optional CSV containing daily observations.")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--target-column", default="supply_m3")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-smoothing", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Small end-to-end validation run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        args.epochs, args.trials, args.runs = 2, 1, 1
    raw = load_supply_csv(args.data, args.date_column, args.target_column) if args.data else make_synthetic_supply(
        days=360 if args.fast else 1_460, seed=args.seed
    )
    cleaned = clean_daily_supply(raw, smoothing=not args.disable_smoothing)
    prepared = prepare_windows(cleaned, args.sequence_length)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("outputs") / stamp; output.mkdir(parents=True, exist_ok=False)
    cleaned.to_csv(output / "cleaned_daily_supply.csv", index=False)
    print(f"Prepared {len(prepared.x_train)} train and {len(prepared.x_test)} test windows. Tuning CNN-LSTM-Attention...")
    tuned, history = tune_cnn_lstm_attention(prepared.x_train, prepared.y_train, args.trials, max(6, args.epochs // 3), args.seed)
    pd.DataFrame(history).to_csv(output / "trial_history.csv", index=False)
    (output / "best_hyperparameters.json").write_text(json.dumps(asdict(tuned), indent=2))
    variants = {"lstm": ModelParams(), "cnn_lstm": ModelParams(), "lstm_attention": ModelParams(), "cnn_lstm_attention": tuned}
    summary, predictions = [], {}
    best_model_to_save = None
    for variant, params in variants.items():
        run_predictions, run_metrics = [], []
        print(f"Training {variant} ({args.runs} run(s))...")
        for run in range(args.runs):
            model, _ = train_model(variant, prepared.x_train, prepared.y_train, prepared.x_test, prepared.y_test, params,
                                   epochs=args.epochs, seed=args.seed + run)
            if variant == "cnn_lstm_attention" and run == 0:
                best_model_to_save = model
            scaled_prediction = predict(model, prepared.x_test)
            actual = prepared.scaler.inverse_transform(prepared.y_test.reshape(-1, 1)).ravel()
            forecast = prepared.scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).ravel()
            run_predictions.append(forecast); run_metrics.append(metrics(actual, forecast))
        predictions[variant] = np.mean(run_predictions, axis=0)
        average = {name: float(np.mean([record[name] for record in run_metrics])) for name in run_metrics[0]}
        summary.append({"model": variant, "runs": args.runs, **average})
    report = pd.DataFrame(summary).sort_values("rmse").reset_index(drop=True)
    report.to_csv(output / "metrics.csv", index=False)
    actual = prepared.scaler.inverse_transform(prepared.y_test.reshape(-1, 1)).ravel()
    save_plots(prepared.test_dates, actual, predictions, report, output)
    config = vars(args); config["data"] = str(args.data) if args.data else "synthetic_generated"
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    save_bundle(output, best_model_to_save, prepared.scaler, tuned, args.sequence_length, args.target_column)
    print("\nHoldout results (lower error is better; higher R² is better):")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nArtifacts saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
