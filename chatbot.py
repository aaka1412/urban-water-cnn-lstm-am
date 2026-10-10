"""Offline terminal chatbot for daily supply or Kharghar billing-period forecasts.

Runs your trained CNN-LSTM-Attention model + history CSV in a chat loop.

Usage:
  python chatbot.py
  python chatbot.py --run outputs\\20260922_151739
  python chatbot.py --run outputs\\20260922_151739 --data data\\navi_mumbai.csv
  python chatbot.py --run outputs\\kharghar_<timestamp>

Try typing:
  tomorrow
  forecast 7 days
  average last 30 days
  explain / explain this        -> plain-language summary of your last answer
  how does the model work
  graph last 30 days / plot forecast 7 days
  graph model comparison
  next billing cycle                 -> for a Kharghar billing-period run
  peak / lowest
  accuracy
  help
  exit
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from water_forecast.data import clean_daily_supply
from water_forecast.inference import find_latest_run, forecast_with_seq_len, load_bundle


HELP = """I can answer (terminal, offline):
  tomorrow | today                  -> next-day forecast in m3
  forecast 7 days / next 14 days    -> multi-day forecast
  average last 30 days              -> history average
  average 2024-05                   -> that month's average (YYYY-MM)
  peak / highest | lowest           -> max/min in known history
  last 7 days                       -> show recent readings
  explain | explain this            -> explain your last answer in plain language
  graph last 30 days | plot forecast 7 days -> save PNG and open it (Windows)
  graph model comparison            -> RMSE chart from training run
  how does the model work           -> short description of CNN-LSTM-Attention
  accuracy / metrics                -> test-set MAE/RMSE/R2 of this run
  dataset                           -> what CSV I need from you (Navi Mumbai)
  help | exit
"""


@dataclass
class ChatContext:
    """Remembers the last answer so 'explain this' can refer to it."""
    last_topic: str | None = None
    last_n: int = 7
    last_forecast_days: int = 1


DATASET_GUIDE = """What dataset I need for Navi Mumbai:
  1 file: data\\navi_mumbai.csv with header: date,supply_m3
  - date: YYYY-MM-DD, one row per day, no gaps ideally (gaps get interpolated).
  - supply_m3: daily TOTAL water supplied in cubic metres (number, e.g. 452310.5).
  - Minimum 120 days, good: 2-4 years (730-1460 rows). More history = better.
  - One series per run: either whole Navi Mumbai total, or one file per zone
    (e.g. data\\nerul.csv, data\\vashi.csv) and train each separately.
  - Keep units consistent (all m3, not litres/MLD mixed). 1 MLD = 1000 m3/day.
  Train: python run_experiment.py --data data\\navi_mumbai.csv
  Then:  python chatbot.py --run <newest outputs folder> --data data\\navi_mumbai.csv
"""

PERIOD_HELP = """I can answer questions about Kharghar billing-period totals:
  forecast / next billing cycle       -> estimate the next two-month total QTY
  forecast 2 periods                  -> estimate the next two billing periods
  average last 6 periods              -> average of recent period totals
  last period / last 4 periods        -> show recent recorded totals
  highest / lowest                    -> period with max/min total QTY
  accuracy                            -> holdout test metrics for this run
  graph last 6 periods                -> save a chart of recent totals
  plot forecast 2 periods             -> save a chart with future estimates
  graph model comparison              -> RMSE comparison across model variants
  how does the model work             -> short explanation
  dataset | help | exit

Each period covers about two months. This chatbot cannot estimate a single month's use.
"""


PERIOD_DATASET_GUIDE = """This Kharghar model uses one total QTY for each complete two-month billing period.
The totals were formed by adding customers' QTY rows for the period. The source unit for QTY
has not yet been confirmed. Ask 'next billing cycle' for the model's next-period estimate.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline terminal chatbot for water-supply forecasts.")
    p.add_argument("--run", type=Path, default=None)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--date-column", default="date")
    p.add_argument("--target-column", default="supply_m3")
    p.add_argument("--no-open-graph", action="store_true", help="Save graphs only; do not open the PNG.")
    return p.parse_args()


def load_period_history(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"period_start", "period_end", "billing_period", "total_qty"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Billing-period CSV is missing {sorted(missing)}. "
            "Use data\\kharghar_period_totals.csv or the run's cleaned_period_totals.csv."
        )
    raw["period_start"] = pd.to_datetime(raw["period_start"], errors="coerce")
    raw["period_end"] = pd.to_datetime(raw["period_end"], errors="coerce")
    raw["total_qty"] = pd.to_numeric(raw["total_qty"], errors="coerce")
    if raw[["period_start", "period_end", "total_qty"]].isna().any().any():
        raise ValueError("Billing-period dates and total_qty must be valid.")
    raw = raw.sort_values("period_start").reset_index(drop=True)
    if raw["billing_period"].duplicated().any() or raw["period_start"].duplicated().any():
        raise ValueError("Billing-period history must contain one row per period.")
    return pd.DataFrame({
        "date": raw["period_start"],
        "period_end": raw["period_end"],
        "billing_period": raw["billing_period"].astype(str),
        "supply_m3": raw["total_qty"],
    })


def load_all(args):
    run_dir = args.run or find_latest_run(Path("outputs"))
    if run_dir is None or not (run_dir / "model_bundle.json").exists():
        print("No trained model bundle found (old runs have no .pt file).")
        print("Run once:  python run_experiment.py --fast")
        print("Then:      python chatbot.py")
        sys.exit(1)
    model, scaler, meta = load_bundle(run_dir)
    period_mode = meta.get("data_frequency") == "two_month_billing_period"
    if period_mode:
        history_path = args.data or (run_dir / "cleaned_period_totals.csv")
        if not history_path.exists():
            raise FileNotFoundError(
                f"No billing-period history found. Provide --data data\\kharghar_period_totals.csv "
                f"or use a run containing cleaned_period_totals.csv: {run_dir}"
            )
        history = load_period_history(history_path)
    elif args.data:
        raw = pd.read_csv(args.data)
        frame = raw[[args.date_column, args.target_column]].rename(
            columns={args.date_column: "date", args.target_column: "supply_m3"})
        history = clean_daily_supply(frame, smoothing=True)
    else:
        history = pd.read_csv(run_dir / "cleaned_daily_supply.csv", parse_dates=["date"])
    metrics = None
    if (run_dir / "metrics.csv").exists():
        metrics = pd.read_csv(run_dir / "metrics.csv")
    return run_dir, model, scaler, meta, history, metrics


def explain_recent(history: pd.DataFrame, n: int) -> str:
    n = max(1, min(len(history), n))
    tail = history.tail(n)
    values = tail["supply_m3"]
    start_d = pd.to_datetime(tail["date"].iloc[0]).date()
    end_d = pd.to_datetime(tail["date"].iloc[-1]).date()
    mean_n = float(values.mean())
    ref = float(history.tail(min(30, len(history)))["supply_m3"].mean())
    delta = mean_n - ref
    trend = float(values.iloc[-1] - values.iloc[0])
    lines = [
        f"These are observed daily totals from your CSV (cleaned/smoothed), not predictions.",
        f"Period: {start_d} to {end_d} ({n} days).",
        f"Average over those days: {mean_n:,.1f} m3/day.",
        f"Highest: {values.max():,.1f} m3 | Lowest: {values.min():,.1f} m3.",
        f"Change from first to last day in the window: {trend:+,.1f} m3.",
    ]
    if abs(ref) > 1e-6:
        pct = 100 * delta / ref
        lines.append(
            f"Compared with the last-30-day average ({ref:,.1f} m3/day): "
            f"{'above' if delta >= 0 else 'below'} by {abs(delta):,.1f} m3/day ({pct:+.1f}%)."
        )
    lines.append("m3 = cubic metres of water supplied that day for the whole area in your file.")
    return "\n".join(lines)


def explain_forecast(model, scaler, meta, history, days: int, metrics: pd.DataFrame | None = None) -> str:
    seq_len = int(meta.get("sequence_length", 30))
    recent = history["supply_m3"].iloc[-seq_len:]
    fc = forecast_with_seq_len(model, scaler, recent, seq_len, days)
    last_date = pd.to_datetime(history["date"].iloc[-1])
    last_actual = float(recent.iloc[-1])
    variant = meta.get("variant", "cnn_lstm_attention")
    lines = [
        f"This is a model forecast, not a meter reading.",
        f"The saved {variant.replace('_', '-')} network looks at the last {seq_len} days of supply, "
        f"then predicts the next day(s). Each extra day reuses the previous prediction as input (roll-forward).",
        f"Last known date: {last_date.date()} at {last_actual:,.1f} m3/day.",
    ]
    if days == 1:
        lines.append(f"Predicted next day: {fc[0]:,.1f} m3 ({fc[0] - last_actual:+,.1f} vs last known).")
    else:
        lines.append(f"Next {days} days average: {fc.mean():,.1f} m3/day (range {fc.min():,.1f}–{fc.max():,.1f}).")
    if metrics is not None and not metrics.empty:
        best = metrics.sort_values("rmse").iloc[0]
        lines.append(
            f"Hold-out test (best model {best['model']}): typical error about RMSE {best['rmse']:,.1f} m3."
        )
    lines.append("Say 'accuracy' for full metrics. Real days can differ from events the model never saw.")
    return "\n".join(lines)


def explain_model(meta: dict, run_dir: Path, metrics: pd.DataFrame | None) -> str:
    variant = meta.get("variant", "cnn_lstm_attention")
    seq = int(meta.get("sequence_length", 30))
    lines = [
        "Offline pipeline (no cloud AI in this chatbot):",
        f"1) Your CSV -> daily series with optional smoothing.",
        f"2) A sliding window of {seq} past days -> neural network ({variant.replace('_', '-').upper()}).",
        "3) CNN layers catch short patterns; LSTM layers catch longer trends; attention weights important days.",
        f"4) Weights loaded from: {run_dir.name}/cnn_lstm_attention.pt",
    ]
    if metrics is not None and not metrics.empty:
        best = metrics.sort_values("rmse").iloc[0]
        lines.append(
            f"On the held-out test split, the best trained variant was {best['model']} "
            f"(RMSE {best['rmse']:,.1f} m3, R2 {best['r2']:.3f}). "
            "Forecasts here still use the saved CNN-LSTM-Attention bundle from training."
        )
    return "\n".join(lines)


def wants_graph(t: str) -> bool:
    return any(w in t for w in ("graph", "chart", "plot", "visuali", "diagram"))


def _open_png(path: Path, open_file: bool) -> str:
    path = path.resolve()
    msg = f"Saved graph:\n  {path}"
    if open_file and sys.platform == "win32":
        os.startfile(path)  # noqa: S606 — intentional for local demo
        msg += "\n(Opened in your default image viewer.)"
    elif open_file:
        msg += "\n(Open that file in your image viewer.)"
    else:
        msg += "\n(Open the path above, or restart without --no-open-graph to auto-open on Windows.)"
    return msg


def plot_recent_history(history: pd.DataFrame, n: int, run_dir: Path, open_file: bool) -> str:
    n = max(1, min(len(history), n))
    tail = history.tail(n).copy()
    tail["date"] = pd.to_datetime(tail["date"])
    out_dir = run_dir / "chatbot_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"history_last{n}.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(tail["date"], tail["supply_m3"], marker="o", linewidth=1.8, color="#1f77b4")
    ax.axhline(tail["supply_m3"].mean(), color="gray", linestyle="--", alpha=0.7, label="period average")
    ax.set(title=f"Daily water supply — last {n} days", ylabel="Supply (m³)", xlabel="Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return _open_png(out, open_file)


def plot_forecast(model, scaler, meta, history, days: int, run_dir: Path, open_file: bool) -> str:
    seq_len = int(meta.get("sequence_length", 30))
    context_n = min(60, len(history))
    hist_tail = history.tail(context_n).copy()
    hist_tail["date"] = pd.to_datetime(hist_tail["date"])
    recent = history["supply_m3"].iloc[-seq_len:]
    fc = forecast_with_seq_len(model, scaler, recent, seq_len, days)
    last_date = pd.to_datetime(history["date"].iloc[-1])
    fut_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days + 1)]
    out_dir = run_dir / "chatbot_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"forecast_next{days}.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist_tail["date"], hist_tail["supply_m3"], label="Recent history", color="black", linewidth=1.6)
    ax.plot(fut_dates, fc, marker="s", label=f"Forecast (+{days}d)", color="#d62728", linewidth=1.8)
    ax.axvline(last_date, color="gray", linestyle=":", alpha=0.8)
    ax.set(title="Water supply: recent history + model forecast", ylabel="Supply (m³)", xlabel="Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return _open_png(out, open_file)


def plot_model_comparison(metrics: pd.DataFrame, run_dir: Path, open_file: bool) -> str:
    existing = run_dir / "model_comparison.png"
    if existing.exists():
        if open_file and sys.platform == "win32":
            os.startfile(existing.resolve())
        return _open_png(existing, False) + "\n(From your training run.)"
    out_dir = run_dir / "chatbot_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model_comparison.png"
    summary = metrics.sort_values("rmse")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    summary.plot.bar(x="model", y="rmse", legend=False, ax=ax, color="#1f77b4")
    ax.set(title="Holdout RMSE by model", xlabel="Model", ylabel="RMSE (m³)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return _open_png(out, open_file)


def next_periods(history: pd.DataFrame, count: int) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    end = pd.to_datetime(history["period_end"].iloc[-1])
    periods = []
    for _ in range(count):
        start = end + pd.Timedelta(days=1)
        end = start + pd.DateOffset(months=2) - pd.Timedelta(days=1)
        label = f"{start.strftime('%b').upper()}-{end.strftime('%b').upper()}{end.year}"
        periods.append((start, end, label))
    return periods


def forecast_period_values(model, scaler, meta, history: pd.DataFrame, count: int) -> np.ndarray:
    seq_len = int(meta.get("sequence_length", 6))
    return forecast_with_seq_len(model, scaler, history["supply_m3"], seq_len, count)


def do_period_forecast(model, scaler, meta, history: pd.DataFrame, count: int) -> str:
    count = max(1, min(12, count))
    forecasts = forecast_period_values(model, scaler, meta, history, count)
    lines = [f"Last recorded period: {history['billing_period'].iloc[-1]} = {history['supply_m3'].iloc[-1]:,.2f} QTY."]
    for (start, end, label), value in zip(next_periods(history, count), forecasts):
        lines.append(f"  Forecast {label} ({start.date()} to {end.date()}): {value:,.2f} QTY.")
    lines.append("This is a model estimate for a two-month billing period; QTY's source unit is unconfirmed.")
    return "\n".join(lines)


def explain_period_forecast(model, scaler, meta, history: pd.DataFrame, metrics: pd.DataFrame | None) -> str:
    seq_len = int(meta.get("sequence_length", 6))
    forecast = forecast_period_values(model, scaler, meta, history, 1)[0]
    next_label = next_periods(history, 1)[0][2]
    lines = [
        "This is a model estimate, not a meter reading.",
        f"The CNN-LSTM-Attention model looks at the last {seq_len} billing periods and estimates the next period.",
        f"Next period ({next_label}) estimate: {forecast:,.2f} QTY.",
        "Each period covers about two months. The source unit for QTY has not been confirmed.",
    ]
    if metrics is not None and not metrics.empty:
        best = metrics.sort_values("rmse").iloc[0]
        lines.append(f"In the holdout check, the lowest RMSE was {best['rmse']:,.2f} QTY ({best['model']}).")
    return "\n".join(lines)


def explain_period_model(meta: dict, run_dir: Path, metrics: pd.DataFrame | None) -> str:
    lines = [
        "This model is trained on Kharghar's combined customer QTY per two-month billing period.",
        f"It uses the last {int(meta.get('sequence_length', 6))} periods to estimate the next period.",
        f"The saved model is in {run_dir.name}/cnn_lstm_attention.pt.",
        "The neural-network architecture is CNN-LSTM-Attention; the time step here means one billing period.",
    ]
    if metrics is not None and not metrics.empty:
        best = metrics.sort_values("rmse").iloc[0]
        lines.append(f"Best holdout RMSE in this run: {best['rmse']:,.2f} QTY ({best['model']}).")
    lines.append("There are few periods, so treat forecasts as exploratory.")
    return "\n".join(lines)


def plot_period_history(history: pd.DataFrame, count: int, run_dir: Path, open_file: bool) -> str:
    count = max(1, min(len(history), count))
    tail = history.tail(count)
    out_dir = run_dir / "chatbot_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"history_last{count}_periods.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(tail["date"], tail["supply_m3"], marker="o", linewidth=1.8, color="#1f77b4")
    axis.set(title=f"Kharghar total QTY — last {count} billing periods",
             ylabel="Total QTY (source unit unconfirmed)", xlabel="Billing period")
    axis.set_xticks(tail["date"], tail["billing_period"], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return _open_png(out, open_file)


def plot_period_forecast(model, scaler, meta, history: pd.DataFrame, count: int,
                         run_dir: Path, open_file: bool) -> str:
    count = max(1, min(6, count))
    forecasts = forecast_period_values(model, scaler, meta, history, count)
    future = next_periods(history, count)
    tail = history.tail(min(8, len(history)))
    labels = list(tail["billing_period"]) + [item[2] for item in future]
    values = list(tail["supply_m3"]) + list(forecasts)
    x = list(range(len(values)))
    out_dir = run_dir / "chatbot_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"forecast_next{count}_periods.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(x[:len(tail)], values[:len(tail)], marker="o", label="Recorded total", color="black")
    axis.plot(x[len(tail)-1:], values[len(tail)-1:], marker="s", label="Forecast", color="#d62728")
    axis.axvline(len(tail) - 0.5, color="gray", linestyle=":", alpha=0.8)
    axis.set(title="Kharghar total QTY by billing period", ylabel="Total QTY (source unit unconfirmed)",
             xlabel="Billing period")
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return _open_png(out, open_file)


def plot_period_model_comparison(metrics: pd.DataFrame, run_dir: Path, open_file: bool) -> str:
    existing = run_dir / "model_comparison.png"
    if existing.exists():
        return _open_png(existing, open_file)
    out_dir = run_dir / "chatbot_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model_comparison.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    summary = metrics.sort_values("rmse")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    summary.plot.bar(x="model", y="rmse", legend=False, ax=axis, color="#1f77b4")
    axis.set(title="Holdout RMSE by model", xlabel="Model", ylabel="RMSE (QTY units)")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return _open_png(out, open_file)


def answer_named_period(text: str, model, scaler, meta, history: pd.DataFrame) -> str | None:
    month_pattern = r"JAN|FEB|MAR|APR|MAY|JUN|JULY|JUL|JLY|AUG|SEP|OCT|NOV|DEC"
    match = re.search(rf"\b({month_pattern})-({month_pattern})(20\d{{2}})\b", text.upper())
    if not match:
        return None

    months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "JULY": 7, "JLY": 7, "AUG": 8, "SEP": 9, "OCT": 10,
              "NOV": 11, "DEC": 12}
    start_month, end_month, end_year = months[match.group(1)], months[match.group(2)], int(match.group(3))
    start_year = end_year - 1 if start_month == 12 else end_year
    target_start = pd.Timestamp(start_year, start_month, 1)
    target_end = target_start + pd.DateOffset(months=2) - pd.Timedelta(days=1)
    requested_label = f"{match.group(1)}-{match.group(2)}{end_year}"
    if target_end.month != end_month or target_end.year != end_year:
        return f"{requested_label} is not a two-month billing period in this dataset."

    recorded = history[pd.to_datetime(history["date"]).dt.normalize() == target_start]
    if not recorded.empty:
        row = recorded.iloc[0]
        return f"Recorded total for {row['billing_period']}: {row['supply_m3']:,.2f} QTY."

    next_start = pd.to_datetime(history["period_end"].iloc[-1]) + pd.Timedelta(days=1)
    month_offset = (target_start.year - next_start.year) * 12 + target_start.month - next_start.month
    if month_offset < 0:
        return f"There is no recorded total for {requested_label} in this run's history."
    if month_offset % 2:
        return f"{requested_label} does not line up with the two-month billing periods in this dataset."
    steps = month_offset // 2 + 1
    if steps > 12:
        return "That period is more than 12 cycles ahead; this chatbot only forecasts up to 12 periods."
    return f"Estimate through {requested_label}:\n" + do_period_forecast(model, scaler, meta, history, steps)


def answer_period(text: str, model, scaler, meta, history, metrics, run_dir, ctx,
                  *, open_graph: bool = True) -> tuple[str, ChatContext]:
    t = text.strip().lower()
    if t in {"exit", "quit", "q"}:
        return "EXIT", ctx
    if t in {"help", "?"}:
        return PERIOD_HELP, ctx
    if wants_graph(t):
        nums = [int(value) for value in re.findall(r"\d+", t)]
        if "comparison" in t or "metric" in t or "model" in t or "accuracy" in t:
            if metrics is None:
                return "No metrics.csv in this run.", ctx
            return plot_period_model_comparison(metrics, run_dir, open_graph), ctx
        if "forecast" in t or "predict" in t or "next" in t:
            return plot_period_forecast(model, scaler, meta, history, nums[0] if nums else 1,
                                        run_dir, open_graph), ctx
        return plot_period_history(history, nums[0] if nums else 6, run_dir, open_graph), ctx
    if "dataset" in t or "what data" in t or "which data" in t:
        return PERIOD_DATASET_GUIDE, ctx
    if any(phrase in t for phrase in ("how does the model", "how do you predict", "how does this work", "what is the model")):
        return explain_period_model(meta, run_dir, metrics), ctx
    if "accur" in t or "metric" in t or "r2" in t or "rmse" in t or "how good" in t:
        if metrics is None or metrics.empty:
            return "No metrics.csv in this run.", ctx
        best = metrics.sort_values("rmse").iloc[0]
        ctx.last_topic = "metrics"
        return (f"Best holdout model: {best['model']} | MAE {best['mae']:,.2f} QTY, "
                f"RMSE {best['rmse']:,.2f} QTY, R² {best['r2']:.3f}. "
                "Lower MAE/RMSE is better; negative R² means this small holdout was difficult to predict."), ctx
    if wants_explanation(t):
        if "forecast" in t or "next" in t or "cycle" in t or ctx.last_topic == "forecast":
            return explain_period_forecast(model, scaler, meta, history, metrics), ctx
        if ctx.last_topic == "average":
            return "That is the average total QTY per two-month billing period, not per day or per month.", ctx
        if ctx.last_topic in {"peak", "lowest"}:
            return "That is the highest or lowest recorded period total, not a forecast.", ctx
        return ("Ask about the next billing cycle, last periods, an average, highest/lowest period, or accuracy; "
                "then say 'explain this'."), ctx
    named_period = answer_named_period(text, model, scaler, meta, history)
    if named_period is not None:
        ctx.last_topic = "forecast" if "Estimate through" in named_period else "recent"
        return named_period, ctx
    if "next month" in t or "tomorrow" in t or "agla mahina" in t or "agle mahine" in t:
        return "The Kharghar data is grouped into two-month billing periods, so it cannot forecast one month by itself. Ask 'next billing cycle'.", ctx
    forecast_request = any(word in t for word in ("forecast", "predict")) or (
        any(word in t for word in ("next", "agla", "agle", "aane wala"))
        and any(word in t for word in ("period", "cycle", "billing"))
    )
    if forecast_request:
        numbers = [int(value) for value in re.findall(r"\d+", t)]
        count = numbers[0] if numbers else 1
        ctx.last_topic = "forecast"
        return do_period_forecast(model, scaler, meta, history, count), ctx
    if "average" in t or "mean" in t or "ausat" in t:
        numbers = [int(value) for value in re.findall(r"\d+", t)]
        count = max(1, min(len(history), numbers[0] if numbers else 6))
        selected = history.tail(count)
        ctx.last_topic = "average"
        return (f"Average total for the last {count} billing periods: "
                f"{selected['supply_m3'].mean():,.2f} QTY."), ctx
    if any(word in t for word in ("highest", "peak", "maximum", "max", "sabse zyada")):
        row = history.loc[history["supply_m3"].idxmax()]
        ctx.last_topic = "peak"
        return f"Highest recorded period total: {row['billing_period']} = {row['supply_m3']:,.2f} QTY.", ctx
    if any(word in t for word in ("lowest", "minimum", "min", "least", "sabse kam")):
        row = history.loc[history["supply_m3"].idxmin()]
        ctx.last_topic = "lowest"
        return f"Lowest recorded period total: {row['billing_period']} = {row['supply_m3']:,.2f} QTY.", ctx
    if any(word in t for word in ("last", "recent", "pichla", "pichle")):
        numbers = [int(value) for value in re.findall(r"\d+", t)]
        count = max(1, min(len(history), numbers[0] if numbers else 1))
        selected = history.tail(count)
        ctx.last_topic, ctx.last_n = "recent", count
        lines = ["Recent billing-period totals:"]
        lines.extend(f"  {row.billing_period}: {row.supply_m3:,.2f} QTY" for row in selected.itertuples())
        return "\n".join(lines), ctx
    if "hello" in t or t == "hi":
        return "Hi. Try 'next billing cycle', 'average last 6 periods', 'highest period', or 'help'.", ctx
    return ("I answer from the Kharghar billing-period model and history. Try 'next billing cycle', "
            "'last 4 periods', 'average last 6 periods', 'accuracy', or 'help'."), ctx


def handle_graph_request(
    t: str, ctx: ChatContext, model, scaler, meta, history, metrics, run_dir: Path, open_file: bool
) -> tuple[str, ChatContext] | None:
    if not wants_graph(t):
        return None
    nums = re.findall(r"\d+", t)
    if "comparison" in t or "metric" in t or "accuracy" in t or "model" in t:
        if metrics is None:
            return "No metrics.csv in this run — train with run_experiment.py first.", ctx
        ctx.last_topic = "graph_metrics"
        return plot_model_comparison(metrics, run_dir, open_file), ctx
    if "forecast" in t or "tomorrow" in t or "predict" in t:
        days = max(1, min(60, int(nums[0]))) if nums else ctx.last_forecast_days
        ctx.last_topic, ctx.last_forecast_days = "graph_forecast", days
        return plot_forecast(model, scaler, meta, history, days, run_dir, open_file), ctx
    if t.strip() in {"graph", "chart", "plot", "graph this", "plot this", "show graph"}:
        if ctx.last_topic == "forecast":
            ctx.last_topic = "graph_forecast"
            return plot_forecast(model, scaler, meta, history, ctx.last_forecast_days, run_dir, open_file), ctx
        if ctx.last_topic == "recent":
            ctx.last_topic = "graph_recent"
            return plot_recent_history(history, ctx.last_n, run_dir, open_file), ctx
        return (
            "Say what to plot, e.g. 'graph last 30 days' or 'plot forecast 7 days' or 'graph model comparison'.",
            ctx,
        )
    n = int(nums[0]) if nums else ctx.last_n
    if "last" in t or "recent" in t or "history" in t or "day" in t:
        n = max(1, min(365, n))
        ctx.last_topic, ctx.last_n = "graph_recent", n
        return plot_recent_history(history, n, run_dir, open_file), ctx
    n = max(7, min(365, n if nums else 30))
    ctx.last_topic, ctx.last_n = "graph_recent", n
    return plot_recent_history(history, n, run_dir, open_file), ctx


def wants_explanation(t: str) -> bool:
    return any(
        phrase in t
        for phrase in (
            "explain",
            "what does this mean",
            "what do these mean",
            "help me understand",
            "make sense",
            "interpret",
            "why is",
            "how to read",
        )
    )


def do_forecast(model, scaler, meta, history, days: int) -> str:
    seq_len = int(meta.get("sequence_length", 30))
    recent = history["supply_m3"].iloc[-seq_len:]
    fc = forecast_with_seq_len(model, scaler, recent, seq_len, days)
    last_date = pd.to_datetime(history["date"].iloc[-1])
    lines = [f"Last known: {last_date.date()} = {recent.iloc[-1]:.1f} m3."]
    for i, v in enumerate(fc, 1):
        day = (last_date + pd.Timedelta(days=i)).date()
        lines.append(f"  +{i}d {day}: {v:.1f} m3")
    if days > 1:
        lines.append(f"Average next {days} days: {fc.mean():.1f} m3/day.")
    return "\n".join(lines)


def answer(
    text: str, model, scaler, meta, history, metrics, run_dir, ctx: ChatContext, *, open_graph: bool = True
) -> tuple[str, ChatContext]:
    if meta.get("data_frequency") == "two_month_billing_period":
        return answer_period(text, model, scaler, meta, history, metrics, run_dir, ctx, open_graph=open_graph)
    t = text.strip().lower()
    if t in {"exit", "quit", "q"}:
        return "EXIT", ctx
    if t in {"help", "?"}:
        return HELP, ctx
    graph_reply = handle_graph_request(t, ctx, model, scaler, meta, history, metrics, run_dir, open_graph)
    if graph_reply is not None:
        return graph_reply
    if "dataset" in t or "navi mumbai" in t or "what data" in t or "which data" in t:
        return DATASET_GUIDE, ctx
    if any(p in t for p in ("how does the model", "how do you predict", "how does this work", "what is the model")):
        return explain_model(meta, run_dir, metrics), ctx
    if wants_explanation(t):
        m = re.search(r"last\s*(\d+)", t)
        if m or ("recent" in t and "day" in t):
            n = int(m.group(1)) if m else ctx.last_n
            return explain_recent(history, n), ctx
        if "forecast" in t or "tomorrow" in t:
            nums = re.findall(r"\d+", t)
            days = max(1, min(60, int(nums[0]))) if nums else ctx.last_forecast_days
            return explain_forecast(model, scaler, meta, history, days, metrics), ctx
        if ctx.last_topic == "recent":
            return explain_recent(history, ctx.last_n), ctx
        if ctx.last_topic == "forecast":
            return explain_forecast(model, scaler, meta, history, ctx.last_forecast_days, metrics), ctx
        if ctx.last_topic == "average":
            return (
                "That number is the mean daily supply (m3/day) over the days you asked for—"
                "useful baseline to compare against a single day or a forecast.",
                ctx,
            )
        if ctx.last_topic == "metrics" and metrics is not None:
            return explain_model(meta, run_dir, metrics), ctx
        if ctx.last_topic in {"peak", "lowest"}:
            return (
                "That value is the highest or lowest single-day total in your full history CSV "
                "(after cleaning), not a forecast.",
                ctx,
            )
        return (
            "I can explain in plain language. Ask something first (e.g. 'last 7 days' or 'tomorrow'), "
            "then say 'explain this'. Or try 'explain last 7 days' / 'how does the model work'.",
            ctx,
        )
    if "accur" in t or "metric" in t or "r2" in t or "rmse" in t or "how good" in t:
        if metrics is None:
            return "No metrics.csv in this run.", ctx
        best = metrics.sort_values("rmse").iloc[0]
        ctx.last_topic = "metrics"
        return (f"Best on hold-out test: {best['model']} | "
                f"MAE {best['mae']:.2f} m3, RMSE {best['rmse']:.2f} m3, R2 {best['r2']:.4f} "
                f"(averaged over {int(best['runs'])} run(s)). Lower error = better, R2 near 1 = better."), ctx
    m = re.search(r"forecast\s*(\d+)?|next\s*(\d+)?\s*days?|(\d+)\s*days?\s*forecast", t)
    if "forecast" in t or "next" in t or "predict" in t or ("consumption" in t and ("tomorrow" in t or "next" in t)):
        days = 7
        nums = re.findall(r"\d+", t)
        if nums:
            days = max(1, min(60, int(nums[0])))
        elif "tomorrow" in t or "today" in t:
            days = 1
        ctx.last_topic, ctx.last_forecast_days = "forecast", days
        return do_forecast(model, scaler, meta, history, days), ctx
    if "tomorrow" in t or ("today" in t and "last" not in t):
        if "explain" not in t:
            ctx.last_topic, ctx.last_forecast_days = "forecast", 1
            return do_forecast(model, scaler, meta, history, 1), ctx
    m = re.search(r"average\s+(\d{4}-\d{2})", t)
    if m:
        ym = m.group(1)
        d = pd.to_datetime(history["date"])
        sel = history[d.dt.strftime("%Y-%m") == ym]
        if sel.empty:
            return f"No data for {ym}. Available: {d.min().date()} to {d.max().date()}.", ctx
        ctx.last_topic = "average"
        return f"Average supply in {ym}: {sel['supply_m3'].mean():.1f} m3/day over {len(sel)} days.", ctx
    m = re.search(r"average.*last\s*(\d+)", t)
    if m or ("average" in t and "mean" in t) or ("average" in t):
        n = int(m.group(1)) if m else 30
        n = max(1, min(len(history), n))
        sel = history.tail(n)
        ctx.last_topic = "average"
        return f"Average of last {n} days: {sel['supply_m3'].mean():.1f} m3/day (up to {pd.to_datetime(history['date'].iloc[-1]).date()}).", ctx
    if "peak" in t or "highest" in t or "maximum" in t or "max" in t:
        i = history["supply_m3"].idxmax()
        ctx.last_topic = "peak"
        return f"Peak in known history: {history.loc[i, 'supply_m3']:.1f} m3 on {pd.to_datetime(history.loc[i, 'date']).date()}.", ctx
    if "lowest" in t or "minimum" in t or "min" in t or "least" in t:
        i = history["supply_m3"].idxmin()
        ctx.last_topic = "lowest"
        return f"Lowest in known history: {history.loc[i, 'supply_m3']:.1f} m3 on {pd.to_datetime(history.loc[i, 'date']).date()}.", ctx
    if "last" in t and "day" in t:
        nums = re.findall(r"\d+", t)
        n = int(nums[0]) if nums else 7
        n = max(1, min(30, n))
        tail = history.tail(n)
        ctx.last_topic, ctx.last_n = "recent", n
        lines = ["Recent readings:"]
        for _, r in tail.iterrows():
            lines.append(f"  {pd.to_datetime(r['date']).date()}: {r['supply_m3']:.1f} m3")
        return "\n".join(lines), ctx
    if "hello" in t or "hi" == t:
        return (
            "Hello. Ask me 'tomorrow', 'forecast 7 days', 'average last 30 days', "
            "'explain this' after any answer, or 'help'.",
            ctx,
        )
    return (
        "I did not understand. I answer from your trained model + history CSV.\n"
        "Try: tomorrow | last 7 days | graph last 30 days | explain this | help",
        ctx,
    )


def main() -> None:
    args = parse_args()
    run_dir, model, scaler, meta, history, metrics = load_all(args)
    if meta.get("data_frequency") == "two_month_billing_period":
        print(f"\nKharghar billing-period chatbot ready (offline). Run: {run_dir.name} | periods: {len(history)}")
        print("Ask about the next billing cycle or type 'help'. Type 'exit' to quit.\n")
    else:
        print(f"\nWater-supply chatbot ready (offline). Run: {run_dir.name} | history rows: {len(history)}")
        print("Ask about consumption or type 'help'. After any answer, try 'explain this'. Type 'exit' to quit.\n")
    ctx = ChatContext()
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nchatbot: bye.")
            break
        if not text:
            continue
        reply, ctx = answer(
            text, model, scaler, meta, history, metrics, run_dir, ctx,
            open_graph=not args.no_open_graph,
        )
        if reply == "EXIT":
            print("chatbot: bye.")
            break
        print("chatbot:\n" + reply + "\n")


if __name__ == "__main__":
    main()
