"""Train the existing CNN/LSTM model family on Kharghar billing-period totals.

One model step is one two-month billing period. This script forecasts the next
billing-period total QTY; it does not make daily forecasts.

Usage:
  python run_kharghar_experiment.py --fast
  python run_kharghar_experiment.py --trials 3 --epochs 50 --runs 1
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent / "src"))
from water_forecast.data import prepare_windows
from water_forecast.inference import save_bundle
from water_forecast.reporting import metrics
from water_forecast.training import ModelParams, predict, train_model
from water_forecast.tuning import tune_cnn_lstm_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a billing-period Kharghar water-use forecaster.")
    parser.add_argument("--data", type=Path, default=Path("data/kharghar_period_totals.csv"))
    parser.add_argument("--sequence-length", type=int, default=6, help="Past billing periods used for each forecast.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="Short workflow check; not a final model run.")
    return parser.parse_args()


def load_period_data(path: Path, sequence_length: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"period_start", "period_end", "billing_period", "total_qty"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    frame["period_start"] = pd.to_datetime(frame["period_start"], errors="coerce")
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame["total_qty"] = pd.to_numeric(frame["total_qty"], errors="coerce")
    if frame[["period_start", "period_end", "total_qty"]].isna().any().any():
        raise ValueError("The period dates and total_qty must all be valid numbers/dates.")
    frame = frame.sort_values("period_start").reset_index(drop=True)
    if frame["billing_period"].duplicated().any() or frame["period_start"].duplicated().any():
        raise ValueError("Each billing period must appear exactly once in the totals CSV.")
    if (frame["total_qty"] < 0).any():
        raise ValueError("Negative total_qty values are not supported.")
    for previous, current in zip(frame["period_start"], frame["period_start"].iloc[1:]):
        if current != previous + pd.DateOffset(months=2):
            raise ValueError(f"Billing periods are not continuous between {previous.date()} and {current.date()}.")
    if len(frame) < sequence_length * 4:
        raise ValueError(
            f"Need at least {sequence_length * 4} periods for sequence_length={sequence_length}; got {len(frame)}."
        )
    return frame


def period_labels(dates: pd.Series, frame: pd.DataFrame) -> list[str]:
    lookup = dict(zip(frame["period_start"], frame["billing_period"]))
    return [lookup[pd.Timestamp(date)] for date in dates]


def save_period_plots(labels: list[str], actual: np.ndarray, predictions: dict[str, np.ndarray],
                      summary: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(positions, actual, marker="o", label="Actual", color="black", linewidth=1.8)
    for name, values in predictions.items():
        axis.plot(positions, values, marker="o", label=name.replace("_", " ").title(), alpha=0.8)
    axis.set(title="Kharghar total QTY: actual vs forecast by billing period",
             ylabel="Total QTY (source unit not confirmed)", xlabel="Billing period")
    axis.set_xticks(positions, labels, rotation=45, ha="right")
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "forecast_vs_actual.png", dpi=180)
    plt.close(fig)

    proposed = predictions["cnn_lstm_attention"]
    fig, axis = plt.subplots(figsize=(5.5, 5.5))
    axis.scatter(actual, proposed, alpha=0.7)
    limits = [min(actual.min(), proposed.min()), max(actual.max(), proposed.max())]
    axis.plot(limits, limits, "r--")
    axis.set(xlabel="Actual total QTY", ylabel="CNN-LSTM-Attention forecast QTY",
             title="Holdout prediction comparison")
    fig.tight_layout()
    fig.savefig(output / "scatter.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    summary.plot.bar(x="model", y="rmse", legend=False, ax=axis, color="#1f77b4")
    axis.set(title="Holdout RMSE by model", xlabel="Model", ylabel="RMSE (QTY units)")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=180)
    plt.close(fig)


def forecast_next_period(model, scaler, history: pd.Series, sequence_length: int) -> float:
    scaled = scaler.transform(pd.DataFrame({"supply_m3": history.to_numpy()})).astype(np.float32).ravel()
    x = torch.tensor(scaled[-sequence_length:], dtype=torch.float32).view(1, sequence_length, 1)
    model.eval()
    with torch.no_grad():
        prediction = model(x).item()
    return float(scaler.inverse_transform(np.array([[prediction]], dtype=np.float64))[0, 0])


def main() -> None:
    args = parse_args()
    if args.fast:
        args.epochs, args.trials, args.runs = 2, 1, 1
    if args.sequence_length < 1 or args.epochs < 1 or args.trials < 1 or args.runs < 1:
        raise ValueError("sequence length, epochs, trials, and runs must be positive.")

    frame = load_period_data(args.data, args.sequence_length)
    model_frame = pd.DataFrame({"date": frame["period_start"], "supply_m3": frame["total_qty"]})
    prepared = prepare_windows(model_frame, args.sequence_length)
    tuned, history = tune_cnn_lstm_attention(
        prepared.x_train, prepared.y_train, args.trials, max(6, args.epochs // 3), args.seed
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("outputs") / f"kharghar_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    period_artifact = frame.copy()
    period_artifact["total_qty"] = period_artifact["total_qty"].astype(float)
    period_artifact.to_csv(output / "cleaned_period_totals.csv", index=False)
    pd.DataFrame(history).to_csv(output / "trial_history.csv", index=False)
    (output / "best_hyperparameters.json").write_text(json.dumps(asdict(tuned), indent=2))

    variants = {
        "lstm": ModelParams(),
        "cnn_lstm": ModelParams(),
        "lstm_attention": ModelParams(),
        "cnn_lstm_attention": tuned,
    }
    summary, predictions = [], {}
    for variant, params in variants.items():
        run_predictions, run_metrics = [], []
        for run in range(args.runs):
            model, _ = train_model(
                variant, prepared.x_train, prepared.y_train, prepared.x_test, prepared.y_test,
                params, epochs=args.epochs, seed=args.seed + run,
            )
            predicted_scaled = predict(model, prepared.x_test)
            actual = prepared.scaler.inverse_transform(prepared.y_test.reshape(-1, 1)).ravel()
            predicted = prepared.scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()
            run_predictions.append(predicted)
            run_metrics.append(metrics(actual, predicted))
        predictions[variant] = np.mean(run_predictions, axis=0)
        average = {name: float(np.mean([record[name] for record in run_metrics])) for name in run_metrics[0]}
        summary.append({"model": variant, "runs": args.runs, **average})

    report = pd.DataFrame(summary).sort_values("rmse").reset_index(drop=True)
    report.to_csv(output / "metrics.csv", index=False)
    actual = prepared.scaler.inverse_transform(prepared.y_test.reshape(-1, 1)).ravel()
    labels = period_labels(prepared.test_dates, frame)
    save_period_plots(labels, actual, predictions, report, output)

    # Refit the selected architecture using every observed period for the next-cycle forecast.
    full_scaler = MinMaxScaler()
    all_scaled = full_scaler.fit_transform(model_frame[["supply_m3"]]).astype(np.float32).ravel()
    x_all = np.asarray([all_scaled[end - args.sequence_length:end] for end in range(args.sequence_length, len(all_scaled))], dtype=np.float32)[..., None]
    y_all = np.asarray([all_scaled[end] for end in range(args.sequence_length, len(all_scaled))], dtype=np.float32)
    final_model, _ = train_model(
        "cnn_lstm_attention", x_all, y_all, x_all, y_all, tuned,
        epochs=args.epochs, seed=args.seed + 10_000,
    )
    predicted_qty = forecast_next_period(final_model, full_scaler, model_frame["supply_m3"], args.sequence_length)
    last_start = frame["period_start"].iloc[-1]
    next_start = last_start + pd.DateOffset(months=2)
    next_end = next_start + pd.DateOffset(months=2) - pd.Timedelta(days=1)
    label = f"{MONTHS_FOR_LABELS[next_start.month]}-{MONTHS_FOR_LABELS[next_end.month]}{next_end.year}"
    pd.DataFrame([{
        "period_start": next_start.strftime("%Y-%m-%d"),
        "period_end": next_end.strftime("%Y-%m-%d"),
        "billing_period": label,
        "forecast_total_qty": round(predicted_qty, 2),
    }]).to_csv(output / "forecast_next_period.csv", index=False)

    save_bundle(output, final_model, full_scaler, tuned, args.sequence_length, target_column="total_qty")
    bundle_path = output / "model_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle.update({"data_frequency": "two_month_billing_period", "target_units": "QTY unit not confirmed"})
    bundle_path.write_text(json.dumps(bundle, indent=2))
    config = vars(args)
    config["data"] = str(args.data)
    config["frequency"] = "two_month_billing_period"
    config["target_column"] = "total_qty"
    config["target_units"] = "QTY unit not confirmed"
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))

    print(f"Prepared {len(prepared.x_train)} training and {len(prepared.x_test)} holdout windows.")
    print("Holdout metrics (QTY units; source unit has not been confirmed):")
    print(report.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"Next-period estimate: {predicted_qty:,.2f} QTY for {label}.")
    print(f"Artifacts saved to: {output.resolve()}")


MONTHS_FOR_LABELS = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JLY", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


if __name__ == "__main__":
    main()
