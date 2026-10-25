"""Train and fairly evaluate the CNN/LSTM model family on Kharghar billing-period totals.

One model step is one two-month billing period. This script forecasts billing-period total QTY;
it does not make daily forecasts.

Evaluation protocol (see src/water_forecast/evaluation.py):
  * walk-forward testing over the last --test-periods bills: for each bill, train only on earlier bills
    and forecast it (plus up to --horizon bills ahead, to check multi-step forecasts);
  * early stopping uses the last training windows, never the test bill;
  * all four neural variants get the same --trials tuning budget on data before the first test bill;
  * naive baselines (last value, same bill last year) are scored on the same bills.

Usage:
  python run_kharghar_experiment.py --fast
  python run_kharghar_experiment.py
  python run_kharghar_experiment.py --trials 20 --runs 5
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

sys.path.insert(0, str(Path(__file__).parent / "src"))
from water_forecast.evaluation import (Fold, fit_on_prefix, horizon_table, identity_preprocess, mean_forecasts,
                                       naive_scale, summarize, tune_all, verdict, walk_forward)
from water_forecast.inference import save_bundle
from water_forecast.reporting import save_comparison_bars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a billing-period Kharghar water-use forecaster.")
    parser.add_argument("--data", type=Path, default=Path("data/kharghar_period_totals.csv"))
    parser.add_argument("--sequence-length", type=int, default=6, help="Past billing periods used for each forecast.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs; early stopping picks fewer.")
    parser.add_argument("--trials", type=int, default=10, help="Tuning trials per model variant (same for all).")
    parser.add_argument("--runs", type=int, default=3, help="Random seeds per model and test bill.")
    parser.add_argument("--test-periods", type=int, default=8, help="Last N bills used as walk-forward test points.")
    parser.add_argument("--horizon", type=int, default=3, help="Also score forecasts up to this many bills ahead.")
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


def save_period_plots(labels: list[str], wide: pd.DataFrame, report: pd.DataFrame, horizons: pd.DataFrame,
                      highlight: str, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    best_baseline = report[report["type"] == "baseline"].iloc[0]["model"]
    shown = list(dict.fromkeys([highlight, "cnn_lstm_attention", best_baseline]))
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(positions, wide["actual"], marker="o", label="Actual", color="black", linewidth=1.8)
    for name in shown:
        style = {"linestyle": "--", "color": "#7f7f7f"} if name == best_baseline else {}
        axis.plot(positions, wide[name], marker="o", label=name.replace("_", " "), alpha=0.85, **style)
    axis.set(title="Kharghar walk-forward test: actual vs forecast (each bill predicted from earlier bills only)",
             ylabel="Total QTY (source unit not confirmed)", xlabel="Billing period")
    axis.set_xticks(positions, labels, rotation=45, ha="right")
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "forecast_vs_actual.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 5.5))
    axis.scatter(wide["actual"], wide[highlight], alpha=0.7)
    limits = [min(wide["actual"].min(), wide[highlight].min()), max(wide["actual"].max(), wide[highlight].max())]
    axis.plot(limits, limits, "r--")
    axis.set(xlabel="Actual total QTY", ylabel=f"{highlight} forecast QTY", title="Walk-forward prediction comparison")
    fig.tight_layout()
    fig.savefig(output / "scatter.png", dpi=180)
    plt.close(fig)

    save_comparison_bars(report, output, "QTY")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for (model, kind), group in horizons.groupby(["model", "type"]):
        style = {"linestyle": "--", "alpha": 0.7} if kind == "baseline" else {"marker": "o"}
        axis.plot(group["horizon"], group["mae"], label=model, **style)
    axis.set(title="Error by forecast horizon (1 = next bill)", xlabel="Bills ahead", ylabel="MAE (QTY)")
    axis.set_xticks(sorted(horizons["horizon"].unique()))
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "horizon_comparison.png", dpi=180)
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
        args.epochs, args.trials, args.runs, args.test_periods, args.horizon = 3, 1, 1, 3, 2
    if min(args.sequence_length, args.epochs, args.trials, args.runs, args.test_periods, args.horizon) < 1:
        raise ValueError("sequence length, epochs, trials, runs, test periods and horizon must be positive.")

    frame = load_period_data(args.data, args.sequence_length)
    raw = frame["total_qty"].to_numpy(dtype=float)
    first_test = len(raw) - args.test_periods
    if first_test - args.sequence_length < 8:
        raise ValueError(f"Too few training windows before the first test bill ({first_test - args.sequence_length}); "
                         "lower --test-periods or --sequence-length.")
    folds = [Fold(train_end=origin, test_indices=(origin,)) for origin in range(first_test, len(raw))]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("outputs") / f"kharghar_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    period_artifact = frame.copy()
    period_artifact["total_qty"] = period_artifact["total_qty"].astype(float)
    period_artifact.to_csv(output / "cleaned_period_totals.csv", index=False)

    print(f"Tuning all 4 variants ({args.trials} trials each) on the {first_test} bills before the test period...")
    tuned, history = tune_all(raw, first_test, args.sequence_length, identity_preprocess, args.trials,
                              max(10, args.epochs // 3), args.seed)
    pd.DataFrame(history).to_csv(output / "trial_history.csv", index=False)
    (output / "best_hyperparameters.json").write_text(
        json.dumps({variant: asdict(params) for variant, params in tuned.items()}, indent=2))

    print(f"Walk-forward test over the last {args.test_periods} bills ({args.runs} seed(s) per model)...")
    predictions = walk_forward(raw, folds, args.sequence_length, tuned, identity_preprocess, args.epochs, args.runs,
                               args.seed, horizon=args.horizon, season=6,
                               progress=lambda message: print("  " + message))
    scale = naive_scale(raw, first_test)
    report = summarize(predictions, scale)
    horizons = horizon_table(predictions, scale)
    report.to_csv(output / "metrics.csv", index=False)
    horizons.to_csv(output / "horizon_metrics.csv", index=False)
    wide = mean_forecasts(predictions)
    wide.insert(1, "billing_period", frame["billing_period"].to_numpy()[wide["target_index"]])
    wide.to_csv(output / "walk_forward_predictions.csv", index=False)

    neural = report[report["type"] == "neural"]
    deployed = neural.iloc[0]["model"]  # deploy the neural variant with the lowest walk-forward error
    save_period_plots(list(wide["billing_period"]), wide, report, horizons, deployed, output)

    # Final model for the chatbot: same leak-free fitting, now on every observed bill.
    final_model, full_scaler, _, final_epochs = fit_on_prefix(
        deployed, tuned[deployed], raw, len(raw), args.sequence_length, identity_preprocess, args.epochs,
        args.seed + 10_000)
    model_frame = pd.DataFrame({"supply_m3": raw})
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

    save_bundle(output, final_model, full_scaler, tuned[deployed], args.sequence_length, target_column="total_qty")
    bundle_path = output / "model_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    best_baseline = report[report["type"] == "baseline"].iloc[0]
    deployed_row = neural.iloc[0]
    bundle.update({
        "data_frequency": "two_month_billing_period",
        "target_units": "QTY unit not confirmed",
        "evaluation": {
            "protocol": f"walk-forward over last {args.test_periods} bills, {args.runs} seed(s), "
                        "early stopping on training windows, equal tuning for all variants",
            "deployed_variant": deployed,
            "deployed_mae": deployed_row["mae"],
            "deployed_mase": deployed_row["mase"],
            "best_baseline": best_baseline["model"],
            "best_baseline_mae": best_baseline["mae"],
            "final_training_epochs": final_epochs,
        },
    })
    bundle_path.write_text(json.dumps(bundle, indent=2))
    config = vars(args)
    config["data"] = str(args.data)
    config["frequency"] = "two_month_billing_period"
    config["target_column"] = "total_qty"
    config["target_units"] = "QTY unit not confirmed"
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))

    columns = ["model", "type", "runs", "n_forecasts", "mae", "rmse", "r2", "rel_mae", "mase", "mae_run_std"]
    print("\nWalk-forward results, next bill (QTY units; source unit not confirmed).")
    print("rel_mae < 1 = better than repeating the last bill on the same test bills; "
          "mae_run_std = spread across seeds.")
    print(report[columns].to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    if args.horizon > 1:
        print("\nMAE by bills ahead (neural models and baselines):")
        print(horizons.pivot_table(index="model", columns="horizon", values="mae")
              .to_string(float_format=lambda value: f"{value:,.0f}"))
    print("\n" + verdict(report))
    print(f"\nDeployed for the chatbot: {deployed} (lowest walk-forward error among neural models), "
          f"retrained on all {len(raw)} bills.")
    print(f"Next-period estimate: {predicted_qty:,.2f} QTY for {label}.")
    print(f"Artifacts saved to: {output.resolve()}")


MONTHS_FOR_LABELS = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JLY", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


if __name__ == "__main__":
    main()
