"""Check whether a CSV is ready for training. Terminal compatible.

Usage:
  python validate_data.py --data data\\navi_mumbai.csv
  python validate_data.py --data data\\navi_mumbai.csv --date-column date --target-column supply_m3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from water_forecast.data import clean_daily_supply, regularize_daily, train_clip_limits


def parse_args():
    p = argparse.ArgumentParser(description="Validate a daily water-supply CSV.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--date-column", default="date")
    p.add_argument("--target-column", default="supply_m3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Checking: {args.data}")
    try:
        raw = pd.read_csv(args.data)
    except Exception as e:
        print(f"FAIL: cannot read CSV: {e}")
        sys.exit(1)
    if args.date_column not in raw.columns or args.target_column not in raw.columns:
        print(f"FAIL: need columns '{args.date_column}' and '{args.target_column}'. Found: {list(raw.columns)}")
        print("Fix: rename header to: date,supply_m3")
        sys.exit(1)
    frame = raw[[args.date_column, args.target_column]].rename(
        columns={args.date_column: "date", args.target_column: "supply_m3"})
    dates = pd.to_datetime(frame["date"], errors="coerce")
    nums = pd.to_numeric(frame["supply_m3"], errors="coerce")
    print(f"Rows: {len(frame)} | bad dates: {int(dates.isna().sum())} | bad numbers: {int(nums.isna().sum())} | "
          f"negatives/zero: {int((nums <= 0).sum())}")
    try:
        # Same steps as run_experiment's training inputs: regularize, clip, causal smoothing.
        daily = regularize_daily(frame)
        limits = train_clip_limits(daily["supply_m3"].to_numpy(), len(daily))
        cleaned = clean_daily_supply(frame, smoothing=True, clip_limits=limits)
    except Exception as e:
        print(f"FAIL: cleaning failed: {e}")
        sys.exit(1)
    span = (pd.to_datetime(cleaned['date'].max()) - pd.to_datetime(cleaned['date'].min())).days + 1
    missing = span - len(cleaned)
    print(f"After cleaning: {len(cleaned)} daily rows from {cleaned['date'].min()} to {cleaned['date'].max()} "
          f"({span} calendar days, {missing} gap-filled).")
    print(f"Range: {cleaned['supply_m3'].min():.1f} .. {cleaned['supply_m3'].max():.1f} m3, mean {cleaned['supply_m3'].mean():.1f} m3.")
    ok = len(cleaned) >= 120
    print("PASS: ready for training." if ok else "FAIL: need at least 120 daily rows (good: 730+).")
    if ok:
        print(f"Next: python run_experiment.py --data {args.data}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
