"""Model access, metrics and evaluation views built from the *saved artifacts* of a run.

Everything numeric here is either (a) produced by the original water_forecast inference code or
(b) recomputed from the run's saved walk-forward predictions. Nothing is trained or invented.
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ROOT = config.ROOT
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from water_forecast.inference import find_latest_run, forecast_future, load_bundle, weights_file  # noqa: E402

_LOCK = threading.Lock()

MODELS = {
    "cnn_lstm_attention": ("CNN-LSTM + Attention", "neural",
                           "1-D convolution over the window, an LSTM, then additive attention over all LSTM steps."),
    "cnn_lstm": ("CNN-LSTM", "neural", "1-D convolution followed by an LSTM; the last LSTM step feeds the output layer."),
    "lstm_attention": ("LSTM + Attention", "neural", "LSTM with additive attention over its steps; no convolution."),
    "lstm": ("LSTM", "neural", "Plain single-layer LSTM; the last step feeds the output layer."),
    "naive_last_value": ("Repeat last value (baseline)", "baseline", "Forecasts the next bill as the previous bill."),
    "seasonal_naive_6": ("Same bill last year (baseline)", "baseline",
                         "Forecasts each bill as the bill six periods (one year) earlier."),
}


class ModelUnavailable(RuntimeError):
    """Raised when the run folder, weights or scaler cannot be loaded."""


def py(obj):
    """Make numpy/pandas values JSON-safe (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ------------------------------------------------------------------ run + model
def resolve_run_dir() -> Path:
    out = config.outputs_dir()
    override = config.run_id_override()
    if override:
        run = out / override
        if not (run / "model_bundle.json").exists():
            raise ModelUnavailable(f"WATER_RUN_ID '{override}' has no model_bundle.json in {out}.")
        return run
    run = find_latest_run(out, frequency="two_month_billing_period")
    if run is None:
        raise ModelUnavailable(f"No billing-period run with a model_bundle.json found in {out}.")
    return run


@lru_cache(maxsize=2)
def _bundle_cached(run_dir: str):
    run = Path(run_dir)
    try:
        model, scaler, meta = load_bundle(run)
    except FileNotFoundError as exc:
        raise ModelUnavailable(f"Missing model artifact: {exc.filename or exc}") from exc
    except Exception as exc:  # corrupt/incompatible weights
        raise ModelUnavailable(f"Could not load the trained model ({type(exc).__name__}: {exc})") from exc
    return model, scaler, meta, run


def get_bundle():
    return _bundle_cached(str(resolve_run_dir()))


def clear_cache() -> None:
    _bundle_cached.cache_clear()


def run_timestamp(run: Path) -> str | None:
    try:
        return datetime.strptime(run.name[-15:], "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        return None


# ------------------------------------------------------------------ metrics
def regression_metrics(actual, predicted) -> dict:
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    n = len(y)
    if n == 0:
        return {"n": 0, "mae": None, "mse": None, "rmse": None, "r2": None, "mape": None, "accuracy": None}
    err = y - p
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    mape = float(np.mean(np.abs(err / y)) * 100) if np.all(y > 0) else None
    return {
        "n": n, "mae": float(np.mean(np.abs(err))), "mse": mse, "rmse": mse ** 0.5,
        "r2": (1 - float(np.sum(err ** 2)) / ss_tot) if n >= 2 and ss_tot > 0 else None,
        "mape": mape, "accuracy": max(0.0, 100 - mape) if mape is not None else None,  # regression accuracy % = 100 - MAPE
    }


def _read(run: Path, name: str) -> pd.DataFrame | None:
    path = run / name
    return pd.read_csv(path) if path.exists() else None


def walk_forward(run: Path) -> pd.DataFrame:
    wf = _read(run, "walk_forward_predictions.csv")
    if wf is None:
        raise ModelUnavailable("walk_forward_predictions.csv is missing from the run folder.")
    return wf


def model_columns(wf: pd.DataFrame) -> list[str]:
    return [m for m in MODELS if m in wf.columns]


def catalog() -> dict:
    """The four neural variants and two baselines with actual walk-forward results."""
    run = resolve_run_dir()
    meta = json.loads((run / "model_bundle.json").read_text())
    wf = walk_forward(run)
    saved = _read(run, "metrics.csv")
    saved = saved.set_index("model") if saved is not None else pd.DataFrame()
    deployed = meta.get("variant")
    has_weights_file = weights_file(run).exists()
    out = []
    for key in model_columns(wf):
        label, kind, desc = MODELS[key]
        m = regression_metrics(wf["actual"], wf[key])
        row = saved.loc[key] if key in saved.index else None
        weights = bool(key == deployed and has_weights_file)
        out.append({
            "key": key, "label": label, "type": kind, "description": desc,
            "has_saved_weights": weights, "can_predict_new_data": weights,
            "deployed": key == deployed,
            "evaluation_status": "Evaluated (walk-forward)" + ("" if weights or kind == "baseline"
                                                              else " - results only, no saved weights"),
            "training_status": ("Final model trained on all periods; weights saved" if weights else
                                "Rule-based, no training" if kind == "baseline" else
                                "Trained per walk-forward fold; weights not saved"),
            "evaluated_at": run_timestamp(run), "n_test_points": m["n"],
            **{k: m[k] for k in ("mae", "mse", "rmse", "r2", "mape", "accuracy")},
            "rel_mae": float(row["rel_mae"]) if row is not None and "rel_mae" in row else None,
            "mase": float(row["mase"]) if row is not None and "mase" in row else None,
            "mae_run_std": float(row["mae_run_std"]) if row is not None and "mae_run_std" in row else None,
        })
    return py({"run_id": run.name, "deployed_variant": deployed, "unit": config.UNIT_LABEL,
               "unit_note": config.UNIT_NOTE, "protocol": meta.get("evaluation", {}).get("protocol"),
               "test_labels": wf["billing_period"].tolist(), "models": out})


def best_model(models: list[dict], metric: str = "rmse") -> dict | None:
    cand = [m for m in models if m.get(metric) is not None]
    if not cand:
        return None
    return max(cand, key=lambda m: m[metric]) if metric == "r2" else min(cand, key=lambda m: m[metric])


def walk_forward_view(model_key: str | None = None) -> dict:
    run = resolve_run_dir()
    wf = walk_forward(run)
    keys = model_columns(wf)
    if model_key and model_key not in keys:
        raise KeyError(model_key)
    series = {k: wf[k].tolist() for k in keys}
    resid = {k: (wf["actual"] - wf[k]).tolist() for k in keys}
    return py({"run_id": run.name, "labels": wf["billing_period"].tolist(), "actual": wf["actual"].tolist(),
               "predictions": series, "residuals": resid, "definition": "residual = actual - predicted",
               "scope": "Held-out walk-forward test bills: each bill was predicted by a model trained only on earlier bills.",
               "unit": config.UNIT_LABEL})


def horizon_table() -> list[dict]:
    t = _read(resolve_run_dir(), "horizon_metrics.csv")
    return py(t.to_dict("records")) if t is not None else []


def forecast_errors(run: Path, variant: str) -> list[float]:
    """Deployed model's walk-forward MAE for 1..k steps ahead (never narrower further out)."""
    t = _read(run, "horizon_metrics.csv")
    if t is None:
        return []
    rows = t[t["model"] == variant].sort_values("horizon")
    return [float(v) for v in np.maximum.accumulate(rows["mae"].to_numpy(float))] if len(rows) else []


# ------------------------------------------------------------------ datasets + predictions
def load_original_frame() -> pd.DataFrame:
    path = config.original_dataset()
    if not path.exists():
        raise ModelUnavailable(f"Original dataset not found at {path}.")
    frame = pd.read_csv(path)
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["period_end"] = pd.to_datetime(frame["period_end"])
    return frame.sort_values("period_start").reset_index(drop=True)


def training_range() -> tuple[float, float]:
    _, scaler, _, _ = get_bundle()
    return float(scaler.data_min_[0]), float(scaler.data_max_[0])


def _next_periods(end: pd.Timestamp, count: int) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    out = []
    for _ in range(count):
        start = end + pd.Timedelta(days=1)
        end = start + pd.DateOffset(months=2) - pd.Timedelta(days=1)
        out.append((start, end, f"{start.strftime('%b').upper()}-{end.strftime('%b').upper()}{end.year}"))
    return out


def _predict_one(model, scaler, window: np.ndarray, seq_len: int) -> float:
    with _LOCK:
        return float(forecast_future(model, scaler, window, 1, seq_len)[0])


def run_prediction(frame: pd.DataFrame, horizon: int, is_original: bool) -> dict:
    """Backtest (one-step predictions on supplied history) + genuine future forecast, using the deployed model."""
    model, scaler, meta, run = get_bundle()
    seq = int(meta["sequence_length"])
    values = frame["total_qty"].to_numpy(float)
    if len(values) < seq:
        raise ValueError(f"Need at least {seq} periods; got {len(values)}.")
    horizon = int(max(1, min(6, horizon)))

    rows = []
    for i in range(seq, len(values)):
        pred = _predict_one(model, scaler, values[i - seq:i], seq)
        rows.append({"label": frame["billing_period"].iloc[i], "actual": float(values[i]), "predicted": pred,
                     "residual": float(values[i] - pred)})
    bt_metrics = regression_metrics([r["actual"] for r in rows], [r["predicted"] for r in rows]) if rows else None

    orig = load_original_frame()
    trained_labels = set(orig["billing_period"].astype(str))
    overlap = int(sum(r["label"] in trained_labels for r in rows))
    if is_original:
        scope = ("In-sample: the deployed model was fitted on every one of these periods, so these errors are "
                 "NOT a test of forecasting skill. Use Model Comparison (walk-forward) for honest accuracy.")
    elif overlap:
        scope = (f"{overlap} of {len(rows)} backtested periods also appear in the data the model was trained on "
                 "(APR-MAY2021 to JUN-JLY2026), so those errors are in-sample, not a held-out test.")
    else:
        scope = "All backtested periods are newer or different from the training data (held-out)."

    with _LOCK:
        fc_vals = forecast_future(model, scaler, values[-seq:], horizon, seq)
    errors = forecast_errors(run, meta["variant"])
    periods = _next_periods(frame["period_end"].iloc[-1], horizon)
    forecast = []
    for step, ((s, e, label), v) in enumerate(zip(periods, fc_vals), start=1):
        err = errors[step - 1] if step <= len(errors) else None
        forecast.append({"step": step, "label": label, "period_start": s.date().isoformat(),
                         "period_end": e.date().isoformat(), "forecast": float(v),
                         "lower": float(v - err) if err is not None else None,
                         "upper": float(v + err) if err is not None else None,
                         "method": "one-step forecast" if step == 1 else
                                   "autoregressive: earlier forecasts are fed back as inputs"})
    return py({
        "model": {"variant": meta["variant"], "label": MODELS.get(meta["variant"], (meta["variant"],))[0],
                  "run_id": run.name, "sequence_length": seq, "evaluated_at": run_timestamp(run)},
        "unit": config.UNIT_LABEL, "unit_note": config.UNIT_NOTE, "horizon": horizon,
        "backtest": {"rows": rows, "metrics": bt_metrics, "scope": scope, "in_sample": is_original or overlap > 0},
        "forecast": forecast,
        "forecast_note": ("Future periods have no actual values yet. Ranges are +/- the deployed model's mean absolute "
                          f"walk-forward error for that many steps ahead; beyond {len(errors)} steps ahead no error was "
                          "measured, so no range is shown."),
        "history": {"labels": frame["billing_period"].tolist(), "values": values.tolist()},
    })


# ------------------------------------------------------------------ views
def series_stats(frame: pd.DataFrame) -> dict:
    v = frame["total_qty"].astype(float)
    i_max, i_min = int(v.idxmax()), int(v.idxmin())
    return {"records": int(len(v)), "total": float(v.sum()), "average": float(v.mean()),
            "max": float(v.max()), "max_label": frame["billing_period"].iloc[i_max],
            "min": float(v.min()), "min_label": frame["billing_period"].iloc[i_min],
            "latest": float(v.iloc[-1]), "latest_label": frame["billing_period"].iloc[-1],
            "first_label": frame["billing_period"].iloc[0]}


def analytics(frame: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict:
    f = frame
    if start:
        f = f[f["period_start"] >= pd.to_datetime(start)]
    if end:
        f = f[f["period_end"] <= pd.to_datetime(end)]
    f = f.reset_index(drop=True)
    if f.empty:
        return {"empty": True, "message": "No periods fall inside the selected date range."}
    v = f["total_qty"].astype(float)
    pct = (v.pct_change() * 100).tolist()
    seasonal: dict[str, list] = {}
    seasons = sorted({int(m) for m in f["period_start"].dt.month})
    for year, grp in f.groupby(f["period_end"].dt.year):
        seasonal[str(year)] = [float(grp.loc[grp["period_start"].dt.month == s, "total_qty"].iloc[0])
                               if (grp["period_start"].dt.month == s).any() else None for s in seasons]
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    yearly = f.groupby(f["period_end"].dt.year)["total_qty"].agg(["sum", "count"]).reset_index()
    return py({"empty": False, "labels": f["billing_period"].tolist(), "values": v.tolist(),
               "pct_change": pct, "rolling_mean_3": v.rolling(3).mean().tolist(),
               "seasonal": {"seasons": [f"{names[s - 1]}-{names[(s) % 12]}" for s in seasons], "series": seasonal},
               "yearly": [{"year": int(r["period_end"]), "total": float(r["sum"]), "periods": int(r["count"])}
                          for r in yearly.to_dict("records")],
               "stats": series_stats(f), "unit": config.UNIT_LABEL, "unit_note": config.UNIT_NOTE})


CATEGORIES = ["low", "normal", "high"]


def confusion(model_key: str, q_low: float = 1 / 3, q_high: float = 2 / 3) -> dict:
    """Derived low/normal/high evaluation on the walk-forward test bills (NOT a native classifier)."""
    if not 0.01 <= q_low < q_high <= 0.99:
        raise ValueError("Quantiles must satisfy 0.01 <= q_low < q_high <= 0.99.")
    run = resolve_run_dir()
    wf = walk_forward(run)
    if model_key not in model_columns(wf):
        raise KeyError(model_key)
    first_test = int(wf["target_index"].min())
    hist = pd.read_csv(run / "cleaned_period_totals.csv")["total_qty"].to_numpy(float)[:first_test]
    t_low, t_high = float(np.quantile(hist, q_low)), float(np.quantile(hist, q_high))

    def cat(x):
        return "low" if x < t_low else ("high" if x > t_high else "normal")

    act = [cat(x) for x in wf["actual"]]
    pre = [cat(x) for x in wf[model_key]]
    matrix = [[sum(1 for a, p in zip(act, pre) if a == ra and p == cp) for cp in CATEGORIES] for ra in CATEGORIES]
    total = len(act)
    correct = sum(matrix[i][i] for i in range(3))
    per_class = []
    for i, c in enumerate(CATEGORIES):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(3)) - tp
        fn = sum(matrix[i]) - tp
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
        per_class.append({"class": c, "support": sum(matrix[i]), "precision": prec, "recall": rec, "f1": f1})
    majority = max(sum(r) for r in matrix) / total if total else None  # always predicting the most common actual class
    present = sum(1 for r in matrix if sum(r) > 0)
    warnings = []
    if present < 2:
        warnings.append("All test bills fall in a single class, so a model that always answers that class would score the "
                        "same accuracy. This accuracy does not show that the model separates low, normal and high consumption.")
    if total < 30:
        warnings.append(f"Only {total} test bills: every percentage here moves by {100 / total:.1f} points per bill.")
    absent = [c["class"] for c in per_class if c["support"] == 0]
    if absent:
        warnings.append("No actual test bill falls in class(es): " + ", ".join(absent) +
                        ". Consumption in the test period is outside the range of earlier bills, so thresholds "
                        "derived from earlier bills put them in one class.")
    return py({
        "kind": "derived_category_classification", "model": model_key, "model_label": MODELS[model_key][0],
        "title": "Consumption-category classification derived from regression outputs",
        "disclaimer": ("The models are regressors. Actual and predicted consumption are mapped to low / normal / high "
                       "with the same thresholds, taken from the quantiles of bills before the first test bill "
                       "(no test data is used). This is not a native classification accuracy."),
        "thresholds": {"low_below": t_low, "high_above": t_high, "q_low": q_low, "q_high": q_high,
                       "fit_on": f"first {first_test} bills (before the test period)"},
        "classes": CATEGORIES, "matrix": matrix, "n": total,
        "accuracy": correct / total if total else None, "majority_class_accuracy": majority,
        "accuracy_informative": present >= 2, "correct": correct, "incorrect": total - correct,
        "per_class": per_class, "warnings": warnings,
        "labels": wf["billing_period"].tolist(), "actual_classes": act, "predicted_classes": pre,
        "regression_metrics": regression_metrics(wf["actual"], wf[model_key]),
    })


def _loss_history(run: Path, deployed: str) -> dict:
    """Per-epoch losses saved by the training script (learning_curves.csv), if this run recorded them."""
    curves = _read(run, "learning_curves.csv")
    if curves is None or curves.empty:
        return {"recorded": False, "message": (
            "Per-epoch training and validation loss were not saved by the training script for this run, so no loss "
            "curves can be drawn. Nothing has been estimated or re-trained to fill the gap.")}
    curves["fold"] = curves["fold"].astype(str)
    final = curves[curves["fold"] == "final"].sort_values("epoch")
    folds = curves[(curves["model"] == deployed) & (curves["fold"] != "final")]
    per_fold = {fold: g.groupby("epoch")[["train_loss", "val_loss"]].mean().reset_index().to_dict("list")
                for fold, g in folds.groupby("fold")}
    best = final.loc[final["val_loss"].idxmin()] if len(final) and final["val_loss"].notna().any() else None
    return {"recorded": True, "unit": "MSE on min-max scaled values",
            "message": ("Training loss is the mean training-batch MSE per epoch; validation loss is the MSE on the "
                        "most recent training windows held out for early stopping. Weights from the best "
                        "validation epoch are kept."),
            "final": final[["epoch", "train_loss", "val_loss"]].to_dict("list"),
            "best_epoch": int(best["epoch"]) if best is not None else None, "folds": per_fold}


def cross_validation_view(run: Path) -> dict:
    """Time-series k-fold CV results (cv_summary.csv, cv_folds.csv, train_test_accuracy.csv) if recorded."""
    summary, folds, acc = (_read(run, n) for n in ("cv_summary.csv", "cv_folds.csv", "train_test_accuracy.csv"))
    if summary is None or folds is None:
        return {"recorded": False}
    per_fold = folds.groupby(["model", "type", "split", "fold"], as_index=False)[
        ["mae", "rmse", "r2", "mape", "accuracy_pct"]].mean()
    bounds = folds.groupby("fold").agg(train_end=("train_end", "first")).reset_index()
    labels = load_original_frame()["billing_period"].tolist()
    test_n = folds[folds["split"] == "test"].groupby("fold")["n"].first()
    fold_info = [{"fold": int(r.fold), "train_bills": int(r.train_end),
                  "test_from": labels[int(r.train_end)] if int(r.train_end) < len(labels) else None,
                  "test_to": labels[min(len(labels), int(r.train_end) + int(test_n[r.fold])) - 1],
                  "test_bills": int(test_n[r.fold])} for r in bounds.itertuples()]
    return {"recorded": True, "n_folds": int(folds["fold"].nunique()),
            "method": ("Time-series k-fold cross-validation (expanding window). Fold k trains on all bills before "
                       "its test block and is tested on the next block; data is never shuffled, so the model never "
                       "learns from bills that come after the ones it is tested on."),
            "folds": fold_info, "summary": summary.to_dict("records"), "per_fold": per_fold.to_dict("records"),
            "accuracy": acc.to_dict("records") if acc is not None else [],
            "accuracy_definition": "Accuracy % = 100 - MAPE (mean absolute percentage error)."}


def training_view() -> dict:
    run = resolve_run_dir()
    model, scaler, meta, _ = get_bundle()
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    trials = _read(run, "trial_history.csv")
    orig = load_original_frame()
    in_sample = run_prediction(orig, 1, True)["backtest"]
    ev = meta.get("evaluation", {})
    loss = _loss_history(run, meta["variant"])
    cv = cross_validation_view(run)
    resid = _read(run, "residual_diagnostics.csv")
    return py({
        "run_id": run.name,
        "loss_history": loss,
        "cross_validation": cv,
        "residual_diagnostics": resid.to_dict("records") if resid is not None else [],
        "training": {
            "deployed_variant": meta["variant"], "training_periods": int(len(orig)),
            "training_windows": int(len(orig) - meta["sequence_length"]),
            "final_training_epochs": ev.get("final_training_epochs"), "hyperparameters": meta["params"],
            "sequence_length": meta["sequence_length"],
            "in_sample_metrics": in_sample["metrics"],
            "in_sample_note": ("Computed now by running the saved weights over the training bills (one-step). "
                               "It describes fit to data the model has seen, not forecasting skill."),
            "duration": None, "duration_note": "Training duration was not recorded.",
        },
        "validation": {
            "method": "Early stopping on the most recent training windows (20%, minimum 2), then refit on all windows.",
            "loss_recorded": loss["recorded"],
            "tuning_trials": trials.to_dict("records") if trials is not None else [],
            "tuning_note": (f"Optuna objective per trial: mean validation loss (scaled MSE) over "
                            f"{cfg.get('cv_folds', 2)}-fold time-series cross-validation."),
            "cross_validation": (f"{cv['n_folds']}-fold time-series cross-validation (expanding window, not shuffled), "
                                 f"plus walk-forward over the last {cfg.get('test_periods', 8)} bills "
                                 f"({cfg.get('runs', 1)} seed(s))." if cv["recorded"] else
                                 f"Walk-forward over the last {cfg.get('test_periods', 8)} bills "
                                 f"({cfg.get('runs', 1)} seed(s)); no k-fold cross-validation was recorded for this run."),
        },
        "testing": {"protocol": ev.get("protocol"), "n_test_points": int(len(walk_forward(run))),
                    "models": catalog()["models"], "horizon_metrics": horizon_table()},
        "config": cfg,
    })


def _status_rank(s: str) -> int:
    return {"failed": 3, "warning": 2, "not_evaluated": 1, "passed": 0}[s]


def validation_panel() -> dict:
    """Real checks only; each carries a status and the reason."""
    checks: list[dict] = []

    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    try:
        model, scaler, meta, run = get_bundle()
        add("Model weights load", "passed", f"{meta['variant']} loaded from {run.name} (weights_only=True).")
    except ModelUnavailable as exc:
        add("Model weights load", "failed", str(exc))
        return {"overall": "failed", "checks": checks}
    orig = load_original_frame()
    saved = _read(run, "forecast_next_period.csv")
    if saved is not None:
        live = float(forecast_future(model, scaler, orig["total_qty"].to_numpy(float)[-meta["sequence_length"]:], 1,
                                     meta["sequence_length"])[0])
        diff = abs(live - float(saved["forecast_total_qty"].iloc[0]))
        add("Saved forecast is reproduced", "passed" if diff < 0.01 else "failed",
            f"Live forecast {live:,.2f} vs saved {float(saved['forecast_total_qty'].iloc[0]):,.2f} (difference {diff:.4f}).")
    else:
        add("Saved forecast is reproduced", "not_evaluated", "forecast_next_period.csv is missing.")
    wf, saved_m = walk_forward(run), _read(run, "metrics.csv")
    if saved_m is not None:
        worst = 0.0
        for _, r in saved_m.iterrows():
            if r["model"] in wf.columns:
                worst = max(worst, abs(regression_metrics(wf["actual"], wf[r["model"]])["rmse"] - r["rmse"]))
        add("Metrics reproduce from saved predictions", "passed" if worst < 1e-3 else "failed",
            f"Largest RMSE difference between metrics.csv and recomputation: {worst:.2e}.")
    cat = catalog()["models"]
    dep = next((m for m in cat if m["deployed"]), None)
    base = best_model([m for m in cat if m["type"] == "baseline"], "mae")
    if dep and base:
        gain = (base["mae"] - dep["mae"]) / base["mae"] * 100
        add("Deployed model beats best baseline (MAE)", "passed" if gain > 0 else "warning",
            f"{dep['mae']:,.0f} vs {base['label']} {base['mae']:,.0f} ({gain:+.1f}%).")
        base_r = best_model([m for m in cat if m["type"] == "baseline"], "rmse")
        g2 = (base_r["rmse"] - dep["rmse"]) / base_r["rmse"] * 100
        add("Deployed model beats best baseline (RMSE)", "passed" if g2 > 5 else "warning",
            f"{dep['rmse']:,.0f} vs {base_r['label']} {base_r['rmse']:,.0f} ({g2:+.1f}%); "
            + ("margin is small." if 0 < g2 <= 5 else ""))
        add("R-squared on test bills", "passed" if (dep["r2"] or -1) > 0 else "warning",
            f"R2 = {dep['r2']:.3f}. " + ("Negative R2 means worse than predicting the test-period mean."
                                          if dep["r2"] < 0 else ""))
        add("Mean percentage error", "passed" if (dep["mape"] or 100) < 5 else "warning",
            f"MAPE = {dep['mape']:.2f}% (accuracy {dep['accuracy']:.2f}%) on {dep['n_test_points']} test bills.")
        add("Test sample size", "passed" if dep["n_test_points"] >= 30 else "warning",
            f"{dep['n_test_points']} walk-forward test bills; results are exploratory (the project README says the same).")
        resid = (wf["actual"] - wf[dep["key"]]).abs() / wf["actual"] * 100
        add("Largest single-bill error", "passed" if resid.max() < 10 else "warning",
            f"Worst bill {wf['billing_period'].iloc[int(resid.idxmax())]}: {resid.max():.1f}% off.")
    loss = _loss_history(run, meta["variant"])
    if loss["recorded"]:
        add("Training/validation loss history", "passed",
            f"Saved per epoch; early stopping kept the weights from epoch {loss['best_epoch']}.")
    else:
        add("Training/validation loss history", "not_evaluated",
            "Not saved by the training script for this run; no curves are shown.")
    cv = cross_validation_view(run)
    if cv["recorded"]:
        acc = {r["model"]: r for r in cv["accuracy"]}.get(meta["variant"])
        test = next((r for r in cv["summary"] if r["model"] == meta["variant"] and r["split"] == "test"), None)
        if test:
            add(f"{cv['n_folds']}-fold cross-validation stability",
                "passed" if test["accuracy_pct_std"] < 5 else "warning",
                f"Test accuracy {test['accuracy_pct_mean']:.2f}% ± {test['accuracy_pct_std']:.2f} across "
                f"{cv['n_folds']} folds (RMSE {test['rmse_mean']:,.0f} ± {test['rmse_std']:,.0f}).")
        if acc:
            gap = acc["train_test_gap"]
            add("Overfitting check (train vs test accuracy)", "passed" if acc["fit_check"] == "ok" else "warning",
                f"Train {acc['train_accuracy']:.2f}% vs CV test {acc['cv_test_accuracy']:.2f}% "
                f"(gap {gap:+.2f} points; flagged above 5).")
    else:
        add("k-fold cross-validation", "not_evaluated", "No cross-validation results were saved for this run.")
    resid = _read(run, "residual_diagnostics.csv")
    if resid is not None and meta["variant"] in set(resid["model"]):
        r = resid.set_index("model").loc[meta["variant"]]
        add("Residual bias", "passed" if r["bias_check"] == "ok" else "warning",
            f"Mean residual {r['mean_residual']:,.0f} {config.UNIT_LABEL} (t = {r['bias_t']:.2f}; |t| > 2 means "
            f"systematic over- or under-prediction); {r['share_under_predicted'] * 100:.0f}% of test bills under-predicted.")
    add("Data-leakage protocol", "not_evaluated",
        "The training code documents causal preprocessing and walk-forward testing and ships pytest tests for it "
        "(tests/test_evaluation.py). The dashboard does not re-verify this.")
    lo, hi = training_range()
    add("Training data range", "passed", f"Model scaler fitted on {lo:,.0f} to {hi:,.0f} {config.UNIT_LABEL}.")
    overall = max((c["status"] for c in checks), key=_status_rank)
    if overall == "not_evaluated":
        overall = "passed"
    return py({"overall": overall, "checks": checks, "run_id": run.name,
               "legend": {"passed": "Check ran and passed", "warning": "Check ran; result needs caution",
                          "failed": "Check ran and failed", "not_evaluated": "Not possible with saved artifacts"}})


def recompute_evaluation() -> dict:
    """Admin 're-evaluate': recompute every metric from saved predictions (no training, no weights touched)."""
    clear_cache()
    cat = catalog()
    return {"run_id": cat["run_id"], "models": [{k: m[k] for k in ("key", "mae", "mse", "rmse", "r2", "mape")}
                                               for m in cat["models"]],
            "validation_overall": validation_panel()["overall"]}


def model_info() -> dict:
    out: dict = {"retraining": {"supported": False, "reason": (
        "Retraining is intentionally not exposed. It would create new weights; this dashboard only reads the "
        "existing trained model. Use run_kharghar_experiment.py from the command line if you really need a new run.")}}
    try:
        _, _, meta, run = get_bundle()
        out.update({"loaded": True, "run_id": run.name, "variant": meta["variant"], "params": meta["params"],
                    "sequence_length": meta["sequence_length"], "target": meta.get("target_column"),
                    "units": meta.get("target_units"), "weights_file": weights_file(run).name,
                    "weights_bytes": weights_file(run).stat().st_size, "evaluated_at": run_timestamp(run)})
    except ModelUnavailable as exc:
        out.update({"loaded": False, "error": str(exc)})
    try:
        runs = sorted(p.name for p in config.outputs_dir().iterdir() if (p / "model_bundle.json").exists())
    except FileNotFoundError:
        runs = []
    out["available_runs"] = runs
    return py(out)
