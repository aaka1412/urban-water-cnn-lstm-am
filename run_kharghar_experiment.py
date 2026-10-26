"""Train and fairly evaluate the CNN/LSTM model family on Kharghar billing-period totals.

One model step is one two-month billing period. This script forecasts billing-period total consumption (in Units);
it does not make daily forecasts.

Evaluation protocol (see src/water_forecast/evaluation.py):
  * walk-forward testing over the last --test-periods bills: for each bill, train only on earlier bills
    and forecast it (plus up to --horizon bills ahead, to check multi-step forecasts);
  * early stopping uses the last training windows, never the test bill;
  * all four neural variants get the same --trials tuning budget on data before the first test bill,
    each trial scored by time-series --cv-folds cross-validation;
  * naive baselines (last value, same bill last year) are scored on the same bills.

Additional model validation (saved next to the walk-forward results):
  * time-series k-fold CV (default 5 folds, expanding window, never shuffled) with train, validation
    and test scores per fold: cv_folds.csv, cv_summary.csv, cv_folds.png;
  * train vs validation vs test accuracy (accuracy % = 100 - MAPE) with an overfitting check:
    train_test_accuracy.csv, train_test_accuracy.png;
  * per-epoch training/validation loss: learning_curves.csv, learning_curves.png;
  * residual analysis of the walk-forward forecasts: residual_diagnostics.csv, residuals.png.

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
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from water_forecast.evaluation import (Fold, accuracy_table, cv_summary, fit_on_prefix, horizon_table,
                                       identity_preprocess, mean_forecasts, naive_scale, residual_diagnostics,
                                       summarize, time_series_cv, tune_all, verdict, walk_forward)
from water_forecast.inference import save_bundle
from water_forecast.reporting import metrics, save_comparison_bars
from water_forecast.training import predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a billing-period Kharghar water-use forecaster.")
    parser.add_argument("--data", type=Path, default=Path("data/kharghar_period_totals.csv"))
    parser.add_argument("--sequence-length", type=int, default=6, help="Past billing periods used for each forecast.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs; early stopping picks fewer.")
    parser.add_argument("--trials", type=int, default=10, help="Tuning trials per model variant (same for all).")
    parser.add_argument("--runs", type=int, default=3, help="Random seeds per model and test bill.")
    parser.add_argument("--test-periods", type=int, default=8, help="Last N bills used as walk-forward test points.")
    parser.add_argument("--horizon", type=int, default=3, help="Also score forecasts up to this many bills ahead.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Time-series cross-validation folds.")
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
             ylabel="Total Units (source unit not confirmed)", xlabel="Billing period")
    axis.set_xticks(positions, labels, rotation=45, ha="right")
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "forecast_vs_actual.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 5.5))
    axis.scatter(wide["actual"], wide[highlight], alpha=0.7)
    limits = [min(wide["actual"].min(), wide[highlight].min()), max(wide["actual"].max(), wide[highlight].max())]
    axis.plot(limits, limits, "r--")
    axis.set(xlabel="Actual total (Units)", ylabel=f"{highlight} forecast (Units)", title="Walk-forward prediction comparison")
    fig.tight_layout()
    fig.savefig(output / "scatter.png", dpi=180)
    plt.close(fig)

    save_comparison_bars(report, output, "Units")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    for (model, kind), group in horizons.groupby(["model", "type"]):
        style = {"linestyle": "--", "alpha": 0.7} if kind == "baseline" else {"marker": "o"}
        axis.plot(group["horizon"], group["mae"], label=model, **style)
    axis.set(title="Error by forecast horizon (1 = next bill)", xlabel="Bills ahead", ylabel="MAE (Units)")
    axis.set_xticks(sorted(horizons["horizon"].unique()))
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "horizon_comparison.png", dpi=180)
    plt.close(fig)


def save_validation_plots(fold_scores: pd.DataFrame, accuracy: pd.DataFrame, curves: pd.DataFrame,
                          wide: pd.DataFrame, labels: list[str], deployed: str, output: Path) -> None:
    """CV per-fold test accuracy, train/validation/test accuracy, learning curves and residuals."""
    plt.style.use("seaborn-v0_8-whitegrid")
    test = fold_scores[fold_scores["split"] == "test"].groupby(["model", "fold"])["accuracy_pct"].mean().unstack(0)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    test.plot(kind="bar", ax=axis, width=0.8)
    axis.set(title="Time-series cross-validation: test accuracy per fold (accuracy = 100 - MAPE)",
             xlabel="Fold (each tests the next block of bills after its training data)", ylabel="Test accuracy (%)")
    axis.set_ylim(max(0.0, float(np.nanmin(test.to_numpy())) - 5), 100)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output / "cv_folds.png", dpi=180)
    plt.close(fig)

    neural = accuracy[accuracy["type"] == "neural"]
    splits = [("train_accuracy", "Train", "#1f77b4"), ("validation_accuracy", "Validation", "#ff7f0e"),
              ("cv_test_accuracy", "CV test", "#2ca02c"), ("walk_forward_test_accuracy", "Walk-forward test", "#7f7f7f")]
    positions, width = np.arange(len(neural)), 0.2
    fig, axis = plt.subplots(figsize=(10, 4.8))
    for i, (column, name, color) in enumerate(splits):
        bars = axis.bar(positions + (i - 1.5) * width, neural[column], width, label=name, color=color)
        axis.bar_label(bars, labels=[f"{v:.1f}" for v in neural[column]], fontsize=7, padding=2)
    axis.set_xticks(positions, neural["model"])
    low = float(np.nanmin(neural[[c for c, _, _ in splits]].to_numpy()))
    axis.set(title="Training vs validation vs testing accuracy (accuracy = 100 - MAPE)", ylabel="Accuracy (%)",
             ylim=(max(0.0, low - 5), 100 + 0.25 * (100 - max(0.0, low - 5))))
    axis.set_yticks([tick for tick in axis.get_yticks() if tick <= 100])  # headroom above 100 is for the legend
    axis.legend(fontsize=8, ncol=4, loc="upper center", frameon=True)
    fig.tight_layout()
    fig.savefig(output / "train_test_accuracy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    final = curves[curves["fold"] == "final"]
    axes[0].plot(final["epoch"], final["train_loss"], label="Training loss")
    axes[0].plot(final["epoch"], final["val_loss"], label="Validation loss")
    best = final.loc[final["val_loss"].idxmin()]
    axes[0].axvline(best["epoch"], color="grey", linestyle=":", label=f"Best epoch ({int(best['epoch'])})")
    axes[0].set(title=f"Deployed model ({deployed}) learning curve", xlabel="Epoch", ylabel="MSE (scaled)")
    axes[0].legend(fontsize=8)
    folds = curves[(curves["model"] == deployed) & (curves["fold"] != "final")]
    for fold, group in folds.groupby("fold"):
        epochs = group.groupby("epoch")[["train_loss", "val_loss"]].mean()
        line, = axes[1].plot(epochs.index, epochs["val_loss"], label=f"Fold {fold} validation")
        axes[1].plot(epochs.index, epochs["train_loss"], color=line.get_color(), linestyle="--", alpha=0.5)
    axes[1].set(title="CV folds (solid = validation, dashed = training)", xlabel="Epoch", ylabel="MSE (scaled)")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)

    residuals = (wide["actual"] - wide[deployed]).to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].bar(labels, residuals, color=["#1f77b4" if r >= 0 else "#d62728" for r in residuals])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title=f"Residuals of {deployed} on walk-forward test bills (actual - forecast)",
                xlabel="Billing period", ylabel="Residual (Units)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].hist(residuals, bins=min(8, len(residuals)), color="#1f77b4", edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(title="Residual distribution", xlabel="Residual (Units)", ylabel="Bills")
    fig.tight_layout()
    fig.savefig(output / "residuals.png", dpi=180)
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
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")

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

    print(f"Tuning all 4 variants ({args.trials} trials each, {args.cv_folds}-fold time-series CV) "
          f"on the {first_test} bills before the test period...")
    tuned, history = tune_all(raw, first_test, args.sequence_length, identity_preprocess, args.trials,
                              max(10, args.epochs // 3), args.seed, cv_folds=args.cv_folds)
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

    print(f"\n{args.cv_folds}-fold time-series cross-validation ({args.runs} seed(s) per model)...")
    fold_scores, curves = time_series_cv(raw, args.sequence_length, tuned, identity_preprocess, args.epochs,
                                         args.seed, n_splits=args.cv_folds, runs=args.runs,
                                         progress=lambda message: print("  " + message))
    cv_table = cv_summary(fold_scores)
    accuracy = accuracy_table(cv_table, report)
    residuals = residual_diagnostics(predictions)
    fold_scores.to_csv(output / "cv_folds.csv", index=False)
    cv_table.to_csv(output / "cv_summary.csv", index=False)
    accuracy.to_csv(output / "train_test_accuracy.csv", index=False)
    residuals.to_csv(output / "residual_diagnostics.csv", index=False)

    # Final model for the chatbot: same leak-free fitting, now on every observed bill.
    final_diag: dict = {}
    final_model, full_scaler, full_inputs, final_epochs = fit_on_prefix(
        deployed, tuned[deployed], raw, len(raw), args.sequence_length, identity_preprocess, args.epochs,
        args.seed + 10_000, diagnostics=final_diag)
    curves = pd.concat([curves, pd.DataFrame(final_diag["history"]).assign(model=deployed, run=0, fold="final")],
                       ignore_index=True)
    curves.to_csv(output / "learning_curves.csv", index=False)
    final_x = np.stack([full_inputs[end - args.sequence_length:end]
                        for end in range(args.sequence_length, len(raw))])[..., None].astype(np.float32)
    final_fit = full_scaler.inverse_transform(predict(final_model, final_x).reshape(-1, 1)).ravel()
    final_train = metrics(raw[args.sequence_length:], final_fit)
    save_validation_plots(fold_scores, accuracy, curves, wide, list(wide["billing_period"]), deployed, output)
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
        "target_units": "Units (physical unit not confirmed)",
        "evaluation": {
            "protocol": f"walk-forward over last {args.test_periods} bills, {args.runs} seed(s), "
                        "early stopping on training windows, equal tuning for all variants",
            "deployed_variant": deployed,
            "deployed_mae": deployed_row["mae"],
            "deployed_mase": deployed_row["mase"],
            "best_baseline": best_baseline["model"],
            "best_baseline_mae": best_baseline["mae"],
            "final_training_epochs": final_epochs,
            "final_train_accuracy_pct": final_train["accuracy_pct"],
            "cv_folds": args.cv_folds,
            "cv_protocol": f"{args.cv_folds}-fold time-series cross-validation (expanding window, not shuffled)",
            "deployed_cv_test_accuracy_pct": float(accuracy.set_index("model").loc[deployed, "cv_test_accuracy"]),
        },
    })
    bundle_path.write_text(json.dumps(bundle, indent=2))
    config = vars(args)
    config["data"] = str(args.data)
    config["frequency"] = "two_month_billing_period"
    config["target_column"] = "total_qty"
    config["target_units"] = "Units (physical unit not confirmed)"
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))

    columns = ["model", "type", "runs", "n_forecasts", "mae", "rmse", "r2", "accuracy_pct", "rel_mae", "mase",
               "mae_run_std"]
    print("\nWalk-forward results, next bill (in Units; physical unit not confirmed).")
    print("rel_mae < 1 = better than repeating the last bill on the same test bills; "
          "mae_run_std = spread across seeds.")
    print(report[columns].to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    if args.horizon > 1:
        print("\nMAE by bills ahead (neural models and baselines):")
        print(horizons.pivot_table(index="model", columns="horizon", values="mae")
              .to_string(float_format=lambda value: f"{value:,.0f}"))
    print("\n" + verdict(report))

    print(f"\n{args.cv_folds}-fold time-series cross-validation, mean ± std across folds "
          "(accuracy = 100 - MAPE):")
    for _, row in cv_table.iterrows():
        print(f"  {row['model']:<20} {row['split']:<10} accuracy {row['accuracy_pct_mean']:6.2f}% "
              f"± {row['accuracy_pct_std']:5.2f}   RMSE {row['rmse_mean']:>12,.0f} ± {row['rmse_std']:,.0f}")
    print("\nTraining vs testing accuracy (%):")
    print(accuracy.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print(f"\nDeployed model trained on all bills: training accuracy {final_train['accuracy_pct']:.2f}% "
          f"(R² {final_train['r2']:.3f}), early stopping chose {final_epochs} epochs.")
    print("\nResidual analysis on walk-forward test bills (|bias_t| > 2 = systematic over/under-prediction):")
    print(residuals.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"\nDeployed for the chatbot: {deployed} (lowest walk-forward error among neural models), "
          f"retrained on all {len(raw)} bills.")
    print(f"Next-period estimate: {predicted_qty:,.2f} Units for {label}.")
    print(f"Artifacts saved to: {output.resolve()}")


MONTHS_FOR_LABELS = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JLY", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


if __name__ == "__main__":
    main()
