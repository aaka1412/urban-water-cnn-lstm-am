import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from water_forecast import evaluation
from water_forecast.baselines import naive_forecasts
from water_forecast.data import causal_savgol, regularize_daily, train_clip_limits
from water_forecast.evaluation import (Fold, accuracy_table, cv_folds, cv_summary, fit_on_prefix, identity_preprocess,
                                       residual_diagnostics, summarize, time_series_cv, walk_forward)
from water_forecast.reporting import metrics
from water_forecast.training import ModelParams


def test_causal_savgol_never_uses_future_values():
    rng = np.random.default_rng(0)
    values = rng.normal(100, 10, 60)
    changed_future = values.copy()
    changed_future[40:] += 1_000
    assert np.allclose(causal_savgol(values)[:40], causal_savgol(changed_future)[:40])


def test_causal_savgol_tracks_a_quadratic_exactly_at_the_newest_point():
    t = np.arange(40, dtype=float)
    quadratic = 0.5 * t**2 - 3 * t + 7
    assert np.allclose(causal_savgol(quadratic, window=15, polyorder=2)[14:], quadratic[14:])


def test_missing_days_take_the_previous_value_not_a_later_one():
    frame = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-05"], "supply_m3": [10.0, 20.0, 99.0]})
    daily = regularize_daily(frame)
    assert daily["supply_m3"].tolist() == [10.0, 20.0, 20.0, 20.0, 99.0]


def test_clip_limits_ignore_the_test_period():
    values = np.concatenate([np.linspace(100, 200, 80), np.full(20, 1e9)])
    assert train_clip_limits(values, 80) == train_clip_limits(values[:80], 80)


def test_naive_baselines_use_only_past_values():
    history = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    forecasts = naive_forecasts(history, origin=8, horizon=2, season=6)
    assert forecasts["naive_last_value"].tolist() == [8.0, 8.0]
    assert forecasts["seasonal_naive_6"].tolist() == [3.0, 4.0]


def test_fold_training_never_sees_test_targets(monkeypatch):
    seq_len, train_end = 4, 30
    raw = np.concatenate([np.linspace(10, 20, train_end), np.full(10, 1e9)])
    seen = {}
    real_fit = evaluation.fit_with_validation

    def spy(variant, x, y, params, epochs, seed, **kwargs):
        seen["windows"], seen["max_input"] = len(y), float(x.max())
        return real_fit(variant, x, y, params, epochs, seed, **kwargs)

    monkeypatch.setattr(evaluation, "fit_with_validation", spy)
    _, scaler, _, _ = fit_on_prefix("lstm", ModelParams(hidden_size=8), raw, train_end, seq_len,
                                    identity_preprocess, epochs=2, seed=0)
    assert seen["windows"] == train_end - seq_len  # only windows whose target is before train_end
    assert scaler.data_max_[0] <= 20 + 1e-9  # the huge test values never reach the scaler
    assert seen["max_input"] <= 1.0 + 1e-6


def test_latest_run_uses_the_timestamp_not_the_folder_prefix(tmp_path):
    from water_forecast.inference import find_latest_run, weights_file

    for name, meta in [("kharghar_20260101_000000", {"data_frequency": "two_month_billing_period"}),
                       ("20260301_000000", {}), ("20260201_000000", {})]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "model_bundle.json").write_text(__import__("json").dumps(meta))
    assert find_latest_run(tmp_path).name == "20260301_000000"
    assert find_latest_run(tmp_path, frequency="daily").name == "20260301_000000"
    assert find_latest_run(tmp_path, frequency="two_month_billing_period").name == "kharghar_20260101_000000"
    (tmp_path / "20260301_000000" / "cnn_lstm_attention.pt").write_bytes(b"")
    assert weights_file(tmp_path / "20260301_000000").name == "cnn_lstm_attention.pt"


def test_walk_forward_scores_neural_models_and_baselines_on_the_same_points():
    raw = np.linspace(100, 160, 40) + np.sin(np.arange(40))
    folds = [Fold(train_end=origin, test_indices=(origin,)) for origin in (36, 37, 38, 39)]
    tuned = {"lstm": ModelParams(hidden_size=8)}
    predictions = walk_forward(raw, folds, 4, tuned, identity_preprocess, epochs=2, runs=1, seed=0,
                               horizon=1, season=6)
    points = predictions.groupby("model")["target_index"].apply(lambda s: sorted(s.unique()))
    assert all(index == [36, 37, 38, 39] for index in points)
    report = summarize(predictions, mase_scale=1.0)
    assert set(report["type"]) == {"neural", "baseline"}
    assert report.loc[report["model"] == "naive_last_value", "rel_mae"].iloc[0] == 1.0


def test_accuracy_is_100_minus_mape():
    result = metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert result["mape"] == pytest.approx(10.0)
    assert result["accuracy_pct"] == pytest.approx(90.0)
    assert metrics(np.array([10.0]), np.array([100.0]))["accuracy_pct"] == 0.0  # floored, never negative


def test_cv_folds_are_chronological_contiguous_and_never_shuffled():
    folds = cv_folds(32, 6, 5)
    assert len(folds) == 5
    assert folds[-1][1] == 32  # the last fold tests the most recent bills
    for (train_end, test_end), (next_train_end, _) in zip(folds, folds[1:]):
        assert train_end < test_end == next_train_end  # test blocks follow each other with no gap or overlap
    assert folds[0][0] - 6 >= 10  # first fold keeps at least 10 training windows


def test_cv_fold_training_never_sees_its_test_block(monkeypatch):
    seq_len = 4
    raw = np.linspace(10, 20, 30)
    seen = []
    real_fit = evaluation.fit_on_prefix

    def spy(variant, params, values, train_end, *args, **kwargs):
        seen.append(train_end)
        return real_fit(variant, params, values, train_end, *args, **kwargs)

    monkeypatch.setattr(evaluation, "fit_on_prefix", spy)
    scores, curves = time_series_cv(raw, seq_len, {"lstm": ModelParams(hidden_size=8)}, identity_preprocess,
                                    epochs=2, seed=0, n_splits=3)
    test = scores[scores["split"] == "test"]
    assert seen == [train_end for train_end, _ in cv_folds(len(raw), seq_len, 3)]
    assert set(test["train_end"]) == set(seen)
    assert set(scores["split"]) == {"train", "validation", "test"}
    assert {"epoch", "train_loss", "val_loss"} <= set(curves.columns) and curves["val_loss"].notna().all()


def test_accuracy_table_flags_overfitting_and_skips_baselines():
    summary = cv_summary(pd.DataFrame([
        {"model": "m", "type": "neural", "run": 0, "fold": 1, "split": s, "mae": 1, "rmse": 1, "r2": 0, "mape": 100 - a,
         "accuracy_pct": a} for s, a in (("train", 99.0), ("validation", 92.0), ("test", 90.0))] + [
        {"model": "naive_last_value", "type": "baseline", "run": 0, "fold": 1, "split": "test", "mae": 1, "rmse": 1,
         "r2": 0, "mape": 5, "accuracy_pct": 95.0}]))
    table = accuracy_table(summary, pd.DataFrame({"model": ["m"], "accuracy_pct": [91.0]})).set_index("model")
    assert table.loc["m", "train_test_gap"] == pytest.approx(9.0)
    assert table.loc["m", "fit_check"] == "possible overfitting"
    assert table.loc["m", "walk_forward_test_accuracy"] == 91.0
    assert table.loc["naive_last_value", "fit_check"].startswith("n/a")


def test_residual_diagnostics_detect_systematic_bias():
    rows = [{"model": "biased", "type": "neural", "run": 0, "origin": i, "horizon": 1, "target_index": i,
             "actual": 100.0 + i, "forecast": 90.0 + i + (i % 2)} for i in range(10)]
    diag = residual_diagnostics(pd.DataFrame(rows)).set_index("model")
    assert diag.loc["biased", "mean_residual"] == pytest.approx(9.5)
    assert diag.loc["biased", "bias_check"] == "biased"
    assert diag.loc["biased", "share_under_predicted"] == 1.0
