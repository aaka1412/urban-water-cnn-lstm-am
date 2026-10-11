import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from water_forecast.data import clean_daily_supply, make_synthetic_supply


def test_synthetic_generator_is_positive_and_daily():
    frame = make_synthetic_supply(days=220, seed=7)
    assert (frame.supply_m3 > 0).all()
    assert frame.date.diff().dropna().dt.days.eq(1).all()


def test_clean_daily_supply_clips_with_the_supplied_limits():
    frame = make_synthetic_supply(days=200, seed=3)
    low, high = float(frame.supply_m3.quantile(0.2)), float(frame.supply_m3.quantile(0.8))
    cleaned = clean_daily_supply(frame, smoothing=False, clip_limits=(low, high))
    assert cleaned["supply_m3"].min() >= low
    assert cleaned["supply_m3"].max() <= high
    assert len(cleaned) == 200


def test_clean_daily_supply_smoothing_is_the_causal_filter():
    frame = make_synthetic_supply(days=200, seed=5)
    smoothed = clean_daily_supply(frame, smoothing=True)
    raw = clean_daily_supply(frame, smoothing=False)
    assert not smoothed["supply_m3"].equals(raw["supply_m3"])
    # The causal filter leaves the earliest days untouched (no past window yet).
    assert smoothed["supply_m3"].iloc[:14].equals(raw["supply_m3"].iloc[:14])
