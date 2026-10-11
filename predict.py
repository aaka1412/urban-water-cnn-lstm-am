"""Terminal forecasting CLI: predict next N days of water supply.

Examples (PowerShell):
  python predict.py --run outputs\\20260922_151739 --days 7
  python predict.py --days 1
  python predict.py --run outputs\\20260922_151739 --data data\\navi_mumbai.csv --days 14
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from water_forecast.inference import (clean_with_bundle_settings, find_latest_run, forecast_future, load_bundle,
                                      run_frequency)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecast future daily water supply (terminal).")
    p.add_argument("--run", type=Path, default=None, help="Run folder with model_bundle.json (default: latest).")
    p.add_argument("--data", type=Path, default=None, help="Optional fresh CSV to take last 30 days from.")
    p.add_argument("--date-column", default="date")
    p.add_argument("--target-column", default="supply_m3")
    p.add_argument("--days", type=int, default=7, help="How many future days to forecast (1-60).")
    p.add_argument("--no-smoothing", action="store_true", help="Skip Savitzky-Golay smoothing on --data input.")
    return p.parse_args()


def load_history(run_dir: Path, args: argparse.Namespace, meta: dict) -> pd.DataFrame:
    if args.data:
        raw = pd.read_csv(args.data)
        if args.date_column not in raw.columns or args.target_column not in raw.columns:
            raise ValueError(f"CSV must contain '{args.date_column}' and '{args.target_column}'. "
                             f"Found: {list(raw.columns)}")
        frame = raw[[args.date_column, args.target_column]].rename(
            columns={args.date_column: "date", args.target_column: "supply_m3"})
        return clean_with_bundle_settings(frame, meta, smoothing_allowed=not args.no_smoothing)
    cleaned_path = run_dir / "cleaned_daily_supply.csv"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"No history found. Provide --data or use a run with cleaned_daily_supply.csv: {run_dir}")
    return pd.read_csv(cleaned_path, parse_dates=["date"])


def main() -> None:
    args = parse_args()
    if not 1 <= args.days <= 60:
        raise ValueError("--days must be between 1 and 60")
    run_dir = args.run or find_latest_run(Path("outputs"), frequency="daily")
    if run_dir is None or not (run_dir / "model_bundle.json").exists():
        print("No trained daily model bundle found.")
        print("Train one first:  python run_experiment.py --fast   (or without --fast)")
        print("Old runs from before this update have no .pt file - retrain once to create it.")
        sys.exit(1)
    if run_frequency(run_dir) != "daily":
        print(f"{run_dir.name} is a billing-period (Kharghar) model; predict.py forecasts days.")
        print("Use:  python chatbot.py --run " + str(run_dir))
        sys.exit(1)
    model, scaler, meta = load_bundle(run_dir)
    seq_len = int(meta.get("sequence_length", 30))
    history = load_history(run_dir, args, meta)
    if len(history) < seq_len:
        raise ValueError(f"Need at least {seq_len} days of history, got {len(history)}")
    recent = history["supply_m3"].iloc[-seq_len:]
    forecast = forecast_future(model, scaler, recent, args.days, seq_len)
    last_date = pd.to_datetime(history["date"].iloc[-1])
    print(f"\nModel: {meta.get('variant')}  |  Run: {run_dir.name}  |  Last known date: {last_date.date()} ({recent.iloc[-1]:.1f} m3)")
    print(f"Next {args.days} day(s) forecast (m3/day):")
    print("-" * 38)
    total = 0.0
    for i, value in enumerate(forecast, 1):
        day = (last_date + timedelta(days=i)).date()
        print(f"  +{i:2d}  {day}  :  {value:10.1f} m3")
        total += float(value)
    print("-" * 38)
    print(f"  Avg: {total / len(forecast):.1f} m3/day   Total: {total:.1f} m3")
    out = run_dir / f"forecast_next{args.days}.csv"
    pd.DataFrame({"date": [(last_date + timedelta(days=i)).date().isoformat() for i in range(1, args.days + 1)],
                  "forecast_m3": forecast.round(2)}).to_csv(out, index=False)
    print(f"Saved to: {out}\n")


if __name__ == "__main__":
    main()
