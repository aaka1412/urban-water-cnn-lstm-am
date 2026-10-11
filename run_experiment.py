"""Train and fairly evaluate CNN-LSTM-Attention and its ablations on a daily supply series.

Evaluation protocol (see src/water_forecast/evaluation.py):
  * the last --test-fraction of days is split into --folds consecutive blocks (walk-forward); each block is
    forecast one day ahead by models trained only on the days before that block;
  * early stopping uses the last training windows, never test days;
  * all four variants get the same --trials tuning budget on days before the first test block;
  * preprocessing is causal: gaps take the last known value, outlier limits come from training days only,
    optional Savitzky-Golay smoothing looks backwards only and is applied to model inputs, not to the truth;
  * forecasts are scored against raw observed values, next to naive baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from water_forecast.data import causal_savgol, load_supply_csv, make_synthetic_supply, regularize_daily, train_clip_limits
from water_forecast.evaluation import (Fold, fit_on_prefix, mean_forecasts, naive_scale, summarize, tune_all, verdict,
                                       walk_forward)
from water_forecast.inference import save_bundle
from water_forecast.reporting import save_plots

SMOOTHING_WINDOW, SMOOTHING_POLYORDER = 15, 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate CNN-LSTM-Attention water-supply forecasters.")
    parser.add_argument("--data", type=Path, help="Optional CSV containing daily observations.")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--target-column", default="supply_m3")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs; early stopping picks fewer.")
    parser.add_argument("--trials", type=int, default=12, help="Tuning trials per model variant (same for all).")
    parser.add_argument("--runs", type=int, default=3, help="Random seeds per model and fold.")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Share of days used for walk-forward testing.")
    parser.add_argument("--folds", type=int, default=3, help="Number of consecutive walk-forward test blocks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-smoothing", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Small end-to-end validation run.")
    return parser.parse_args()


def make_preprocess(smoothing: bool):
    """Causal preprocessing: outlier limits from training days, backward-looking smoothing on inputs only."""
    def preprocess(raw: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(raw, *train_clip_limits(raw, train_end))
        inputs = causal_savgol(clipped, SMOOTHING_WINDOW, SMOOTHING_POLYORDER) if smoothing else clipped
        return inputs, clipped
    return preprocess


def main() -> None:
    args = parse_args()
    if args.fast:
        args.epochs, args.trials, args.runs, args.folds = 2, 1, 1, 1
    if not 0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction must be between 0 and 0.5")
    raw_frame = load_supply_csv(args.data, args.date_column, args.target_column) if args.data else make_synthetic_supply(
        days=360 if args.fast else 1_460, seed=args.seed
    )
    daily = regularize_daily(raw_frame)
    raw = daily["supply_m3"].to_numpy(dtype=float)
    first_test = int(len(raw) * (1 - args.test_fraction))
    if first_test < args.sequence_length * 4:
        raise ValueError("Not enough daily records before the test period for the selected sequence length")
    blocks = np.array_split(np.arange(first_test, len(raw)), args.folds)
    folds = [Fold(train_end=int(block[0]), test_indices=tuple(int(i) for i in block)) for block in blocks]
    preprocess = make_preprocess(smoothing=not args.disable_smoothing)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("outputs") / stamp; output.mkdir(parents=True, exist_ok=False)
    print(f"{len(raw)} days; walk-forward test on the last {len(raw) - first_test} days in {args.folds} block(s).")
    print(f"Tuning all 4 variants ({args.trials} trials each) on days before the test period...")
    tuned, history = tune_all(raw, first_test, args.sequence_length, preprocess, args.trials,
                              max(6, args.epochs // 3), args.seed)
    pd.DataFrame(history).to_csv(output / "trial_history.csv", index=False)
    (output / "best_hyperparameters.json").write_text(
        json.dumps({variant: asdict(params) for variant, params in tuned.items()}, indent=2))

    predictions = walk_forward(raw, folds, args.sequence_length, tuned, preprocess, args.epochs, args.runs, args.seed,
                               horizon=1, ma_window=7, season=7, progress=lambda message: print("  " + message))
    report = summarize(predictions, naive_scale(raw, first_test))
    report.to_csv(output / "metrics.csv", index=False)
    wide = mean_forecasts(predictions)
    wide.insert(1, "date", daily["date"].to_numpy()[wide["target_index"]])
    wide.to_csv(output / "walk_forward_predictions.csv", index=False)

    neural = report[report["type"] == "neural"]
    deployed = neural.iloc[0]["model"]
    best_baseline = report[report["type"] == "baseline"].iloc[0]["model"]
    shown = {name: wide[name].to_numpy() for name in dict.fromkeys([deployed, "cnn_lstm_attention", best_baseline])}
    save_plots(pd.Series(wide["date"]), wide["actual"].to_numpy(), shown, report, output, deployed)

    # Final model for predict.py / chatbot.py: same causal preprocessing, fitted on every observed day.
    limits = train_clip_limits(raw, len(raw))
    final_model, scaler, _, final_epochs = fit_on_prefix(
        deployed, tuned[deployed], raw, len(raw), args.sequence_length, preprocess, args.epochs, args.seed + 10_000)
    history_frame = daily.copy()
    history_frame["supply_m3"] = preprocess(raw, len(raw))[0]  # the inputs the model expects, not the raw truth
    history_frame.to_csv(output / "cleaned_daily_supply.csv", index=False)
    config = vars(args); config["data"] = str(args.data) if args.data else "synthetic_generated"
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    save_bundle(output, final_model, scaler, tuned[deployed], args.sequence_length, args.target_column)
    bundle_path = output / "model_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle.update({
        "preprocessing": {"clip_limits": list(limits), "smoothing": not args.disable_smoothing,
                          "window": SMOOTHING_WINDOW, "polyorder": SMOOTHING_POLYORDER},
        "evaluation": {"protocol": f"walk-forward, {args.folds} block(s) over last {args.test_fraction:.0%} of days, "
                                   f"{args.runs} seed(s), early stopping on training windows, equal tuning",
                       "deployed_variant": deployed, "best_baseline": best_baseline,
                       "final_training_epochs": final_epochs},
    })
    bundle_path.write_text(json.dumps(bundle, indent=2))

    columns = ["model", "type", "runs", "n_forecasts", "mae", "rmse", "r2", "rel_mae", "mase", "mae_run_std"]
    print("\nWalk-forward results vs raw values (rel_mae < 1 = better than repeating yesterday on the same days):")
    print(report[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n" + verdict(report))
    print(f"\nDeployed for predict.py / chatbot.py: {deployed}, retrained on all {len(raw)} days.")
    print(f"Artifacts saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
