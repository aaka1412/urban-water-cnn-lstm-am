import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from water_forecast.data import clean_daily_supply, make_synthetic_supply, prepare_windows


def test_pipeline_creates_chronological_windows_without_scale_leakage():
    frame = make_synthetic_supply(days=240, seed=1)
    prepared = prepare_windows(clean_daily_supply(frame, smoothing=True), sequence_length=30)
    assert prepared.x_train.shape[1:] == (30, 1)
    assert len(prepared.x_train) > len(prepared.x_test) > 0
    assert prepared.test_dates.is_monotonic_increasing


def test_synthetic_generator_is_positive_and_daily():
    frame = make_synthetic_supply(days=220, seed=7)
    assert (frame.supply_m3 > 0).all()
    assert frame.date.diff().dropna().dt.days.eq(1).all()
