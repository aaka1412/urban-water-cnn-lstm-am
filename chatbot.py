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
  explain the graph / trend          -> plain-language reading of the last graph
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Graphs are only saved to PNG files, never shown in a window. A file-only backend avoids
# Tk/GUI errors that can crash a terminal chat session.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from water_forecast.inference import (clean_with_bundle_settings, find_latest_run, forecast_future, load_bundle,
                                      weights_file)


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
  explain the graph | trend         -> simple reading of the last graph (up/down, high/low)
  how does the model work          -> short description of CNN-LSTM-Attention
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
    last_graph_kind: str | None = None
    # Rebuilds the last graph's reading in a given language ("en" or "hi"), for 'explain the graph'.
    last_graph_describe: Callable[[str], str] | None = None
    lang_mode: str = "auto"  # "auto" follows the user's wording; "en" / "hi" are fixed
    lang: str = "en"  # language of the current reply


HINGLISH_WORDS = {
    "kya", "hai", "hain", "batao", "bataiye", "bolo", "samjhao", "samjhaao", "samjha", "samjhana", "samjhaiye",
    "agla", "agle", "aane", "pichla", "pichle", "kitna", "kitni", "kitne", "mein", "dikhao", "hoga", "hogi",
    "rahega", "rahegi", "mahina", "mahine", "sabse", "zyada", "jyada", "ka", "ki", "ke", "aur", "bhi", "ko",
    "se", "matlab", "ausat", "anuman", "paani", "kaisa", "kaise", "kab", "wala", "wale", "chahiye", "nahi",
    "namaste", "dijiye", "kijiye",
}


def detect_lang(t: str, mode: str) -> str:
    if mode in {"en", "hi"}:
        return mode
    return "hi" if set(re.findall(r"[a-z]+", t)) & HINGLISH_WORDS else "en"


def set_language(t: str, ctx: ChatContext) -> str | None:
    if t in {"hinglish", "hindi", "hinglish mein", "hindi mein", "hinglish me", "hindi me",
             "hinglish mein bolo", "hindi mein bolo", "hinglish please"}:
        ctx.lang_mode = "hi"
        return "Theek hai! Ab se main Hinglish mein jawab dunga. English ke liye 'english' likhiye."
    if t in {"english", "in english", "english please", "english mein", "english me"}:
        ctx.lang_mode = "en"
        return "Okay, I will reply in English from now on. Type 'hinglish' to switch back."
    if t in {"auto", "auto language", "language auto"}:
        ctx.lang_mode = "auto"
        return "Auto mode: I reply in Hinglish when you write in Hinglish, otherwise in English."
    return None


def _t(lang: str, en: str, hi: str) -> str:
    return hi if lang == "hi" else en


def _fmt(value: float, lang: str, decimals: int, signed: bool = False) -> str:
    """Hinglish replies show big numbers in lakh (42.40 lakh); English keeps full digits."""
    sign = "+" if signed else ""
    if lang == "hi" and abs(value) >= 1e5:
        return f"{value / 1e5:{sign},.2f} lakh"
    if lang == "hi" and abs(value) >= 100:
        return f"{value:{sign},.0f}"  # whole numbers read easier below one lakh
    return f"{value:{sign},.{decimals}f}"


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
  explain the graph / trend           -> simple reading of the last graph (up/down, high/low)
  nov-dec / oct-nov2026               -> estimate or recorded total for any two months
  how does the model work             -> short explanation
  hinglish | english | auto           -> reply language (auto = follow how you write)
  dataset | help | exit

Each period covers about two months. This chatbot cannot estimate a single month's use.
"""

PERIOD_HELP_HI = """Main Kharghar ke billing-period totals ke baare mein bata sakta hoon:
  agla bill / next billing cycle      -> agle do mahine ke bill ka anuman
  agle 2 bills ka anuman              -> agle do bills ka anuman
  nov-dec / oct-nov2026               -> kisi bhi do mahino ka anuman ya recorded total
  pichle 6 bills ka average           -> haal ke bills ka average
  pichle 4 bills                      -> haal ke recorded totals
  sabse zyada / sabse kam             -> sabse bada ya sabse chhota bill
  accuracy kitni hai                  -> model kitna sahi hai (holdout test)
  graph last 6 periods                -> haal ke bills ka graph
  plot forecast 2 periods             -> anuman ke saath graph
  graph ka trend samjhao              -> pichle graph ko aasaan bhasha mein samjhao
  english | hinglish | auto           -> jawab ki bhasha
  help | exit

Har bill lagbhag do mahine ka hota hai, isliye sirf ek mahine ka anuman nahi ho sakta.
"""


PERIOD_DATASET_GUIDE = """This Kharghar model uses one total QTY for each complete two-month billing period.
The totals were formed by adding customers' QTY rows for the period. The source unit for QTY
has not yet been confirmed. Ask 'next billing cycle' for the model's next-period estimate.
"""

PERIOD_DATASET_GUIDE_HI = """Yeh Kharghar model har poore do-mahine ke bill ka ek total QTY use karta hai.
Yeh total us period ke sabhi customers ki QTY jod kar banaya gaya hai. QTY ki unit abhi confirm
nahi hai. Agle bill ka anuman jaanne ke liye 'agla bill' poochiye.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline terminal chatbot for water-supply forecasts.")
    p.add_argument("--run", type=Path, default=None)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--date-column", default="date")
    p.add_argument("--target-column", default="supply_m3")
    p.add_argument("--no-open-graph", action="store_true", help="Save graphs only; do not open the PNG.")
    p.add_argument("--lang", choices=["auto", "en", "hinglish"], default="auto",
                   help="Reply language. auto = Hinglish when you write in Hinglish, otherwise English.")
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
    if args.run is not None and not args.run.is_dir():
        print(f"Run folder not found: {args.run}")
        runs = sorted(p.name for p in Path("outputs").glob("*") if (p / "model_bundle.json").exists())
        if runs:
            print("Available runs:\n  " + "\n  ".join(f"outputs\\{name}" for name in runs))
        print("Tip: type 'outputs\\kh' and press Tab to auto-complete the folder name.")
        sys.exit(1)
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
        history = clean_with_bundle_settings(frame, meta)
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
    fc = forecast_future(model, scaler, recent, days, seq_len)
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
        lines.append(accuracy_text(metrics, meta, "m3", 1, "en"))
    lines.append("Real days can differ from events the model never saw.")
    return "\n".join(lines)


def explain_model(meta: dict, run_dir: Path, metrics: pd.DataFrame | None) -> str:
    variant = meta.get("variant", "cnn_lstm_attention")
    seq = int(meta.get("sequence_length", 30))
    lines = [
        "Offline pipeline (no cloud AI in this chatbot):",
        "1) Your CSV -> daily series; gaps take the last known value, smoothing looks only backwards.",
        f"2) A sliding window of {seq} past days -> neural network ({variant.replace('_', '-').upper()}).",
        "3) CNN layers catch short patterns; LSTM layers catch longer trends; attention weights important days.",
        f"4) Weights loaded from: {run_dir.name}/{weights_file(run_dir).name} (the saved variant is {variant}).",
    ]
    if metrics is not None and not metrics.empty:
        lines.append(accuracy_text(metrics, meta, "m3", 1, "en"))
    return "\n".join(lines)


def accuracy_text(metrics: pd.DataFrame, meta: dict, unit: str, decimals: int, lang: str) -> str:
    """Accuracy of the model this chatbot uses, next to the best simple baseline on the same test points."""
    def f(value: float) -> str:
        return _fmt(float(value), lang, decimals)

    if "type" not in metrics:  # runs made before walk-forward evaluation: single split, test-set early stopping
        best = metrics.sort_values("rmse").iloc[0]
        return _t(lang,
                  f"Best holdout model: {best['model']} | MAE {f(best['mae'])} {unit}, RMSE {f(best['rmse'])} {unit}. "
                  "Note: this older run used a single test split and is likely optimistic; retrain for fair numbers.",
                  f"Sabse accha holdout model: {best['model']} | MAE {f(best['mae'])} {unit}, RMSE {f(best['rmse'])} {unit}. "
                  "Dhyan dein: yeh purana run hai (ek hi test split), iske numbers zyada acche dikh sakte hain; "
                  "sahi numbers ke liye dobara train karein.")
    neural = metrics[metrics["type"] == "neural"]
    used = neural[neural["model"] == meta.get("variant")]
    row = (used if not used.empty else neural.sort_values("mae")).iloc[0]
    base = metrics[metrics["type"] == "baseline"].sort_values("mae").iloc[0]
    gap = 100 * (base["mae"] - row["mae"]) / base["mae"]
    if gap > 2:
        verdict_en = f"The model is {gap:.0f}% more accurate than this simple baseline."
        verdict_hi = f"Model is simple tareeke se {gap:.0f}% zyada sahi hai."
    elif gap < -2:
        loss = 100 * (row["mae"] - base["mae"]) / row["mae"]
        verdict_en = (f"The simple baseline is {loss:.0f}% more accurate than the model, "
                      "so treat the model's forecasts with care.")
        verdict_hi = (f"Simple tareeka model se {loss:.0f}% zyada sahi hai, "
                      "isliye model ke anuman ko saavdhani se dekhein.")
    else:
        verdict_en = "The model and the simple baseline are about equally accurate."
        verdict_hi = "Model aur simple tareeka lagbhag barabar sahi hain."
    return _t(
        lang,
        "Walk-forward test (each point forecast only from earlier data):\n"
        f"  Model used here: {row['model']} | MAE {f(row['mae'])} {unit}, RMSE {f(row['rmse'])} {unit}, "
        f"relMAE {row['rel_mae']:.2f}.\n"
        f"  Best simple baseline: {base['model']} | MAE {f(base['mae'])} {unit}, relMAE {base['rel_mae']:.2f}.\n"
        f"  {verdict_en}\n"
        "  (relMAE below 1 = better than simply repeating the last value.)",
        "Walk-forward test (har point sirf pehle ke data se predict kiya gaya):\n"
        f"  Yahan use hone wala model: {row['model']} | MAE {f(row['mae'])} {unit}, RMSE {f(row['rmse'])} {unit}, "
        f"relMAE {row['rel_mae']:.2f}.\n"
        f"  Sabse accha simple tareeka: {base['model']} | MAE {f(base['mae'])} {unit}, relMAE {base['rel_mae']:.2f}.\n"
        f"  {verdict_hi}\n"
        "  (relMAE 1 se kam = pichla value dohraane se behtar.)",
    )


def wants_graph(t: str) -> bool:
    return any(w in t for w in ("graph", "chart", "plot", "visuali", "diagram"))


def wants_trend(t: str) -> bool:
    return any(
        w in t
        for w in ("trend", "increasing", "decreasing", "going up", "going down", "badh raha", "ghat raha")
    )


def _open_png(path: Path, open_file: bool, lang: str = "en") -> str:
    path = path.resolve()
    msg = _t(lang, "Saved graph:", "Graph save ho gaya:") + f"\n  {path}"
    if open_file and sys.platform == "win32":
        try:
            os.startfile(path)  # noqa: S606 — intentional for local demo
            msg += _t(lang, "\n(Opened in your default image viewer.)", "\n(Image viewer mein khol diya hai.)")
        except OSError as exc:
            msg += _t(lang, f"\n(Could not open it automatically: {exc}. Open the path above yourself.)",
                      f"\n(Apne-aap nahi khul paaya: {exc}. Upar wala path khud kholiye.)")
    elif open_file:
        msg += _t(lang, "\n(Open that file in your image viewer.)", "\n(Yeh file image viewer mein kholiye.)")
    else:
        msg += _t(lang, "\n(Open the path above, or restart without --no-open-graph to auto-open on Windows.)",
                  "\n(Upar wala path kholiye, ya --no-open-graph ke bina chalaiye taaki graph apne-aap khule.)")
    return msg


DIRECTION_TEXT = {
    "up": ("going UP", "UPAR ja raha hai"),
    "down": ("going DOWN", "NEECHE ja raha hai"),
    "flat": ("roughly FLAT (no clear rise or fall)", "lagbhag SEEDHA hai (na khaas badha, na ghata)"),
    "up_slow": ("going UP slowly", "dheere-dheere UPAR ja raha hai"),
    "down_slow": ("going DOWN slowly", "dheere-dheere NEECHE ja raha hai"),
}
STEP_TEXT = {"billing periods": "bills", "days": "dinon"}


def _direction(pct: float, threshold: float) -> str:
    if pct > threshold:
        return "up"
    if pct < -threshold:
        return "down"
    return "flat"


def _dir_text(key: str, lang: str) -> str:
    en, hi = DIRECTION_TEXT[key]
    return _t(lang, en, hi)


def _pct_text(change: float, base: float) -> str:
    return f", {100 * change / base:+.1f}%" if abs(base) > 1e-9 else ""


def describe_trend(values, labels, unit: str, step: str, decimals: int = 1, lang: str = "en") -> str:
    """Plain-language reading of a history line graph: direction, start vs end, high/low, biggest moves."""
    vals = np.asarray(list(values), dtype=float)
    labels = [str(label) for label in labels]

    def f(value: float, signed: bool = False) -> str:
        return _fmt(value, lang, decimals, signed)

    header = _t(lang, "How to read this graph:", "Graph kaise padhein:")
    if len(vals) < 2:
        return header + _t(
            lang,
            f"\n  Only one point ({labels[0]}: {f(vals[0])} {unit}), so there is no trend to read yet. "
            "Ask for more points, e.g. 'graph last 6'.",
            f"\n  Sirf ek point hai ({labels[0]}: {f(vals[0])} {unit}), isliye abhi koi trend nahi dikhta. "
            "Zyada points maangiye, jaise 'graph last 6'.",
        )
    n = len(vals)
    mean = float(vals.mean())
    # Straight-line fit over the whole window, so one odd point does not decide the direction.
    fitted_change = float(np.polyfit(np.arange(n), vals, 1)[0]) * (n - 1)
    fitted_pct = 100 * fitted_change / abs(mean) if abs(mean) > 1e-9 else 0.0
    change = vals[-1] - vals[0]
    steps = np.diff(vals)
    hi, lo = int(vals.argmax()), int(vals.argmin())
    direction = _direction(fitted_pct, 5.0)
    # A small but steady move (most steps the same way) is still a trend, just a slow one.
    if direction == "flat" and fitted_pct > 1 and (steps > 0).mean() >= 0.75:
        direction = "up_slow"
    elif direction == "flat" and fitted_pct < -1 and (steps < 0).mean() >= 0.75:
        direction = "down_slow"
    up, down = int(steps.argmax()), int(steps.argmin())
    median = float(np.median(vals))
    outlier = n >= 4 and median > 0 and vals[hi] > 1.5 * median
    if lang == "hi":
        lines = [
            header,
            f"  Kul mila kar: in {n} {STEP_TEXT.get(step, step)} mein trend {_dir_text(direction, lang)}.",
            f"  Shuru -> ant: {labels[0]} {f(vals[0])} -> {labels[-1]} {f(vals[-1])} {unit} "
            f"({f(change, True)}{_pct_text(change, vals[0])}).",
            f"  Sabse zyada: {labels[hi]} ({f(vals[hi])}) | Sabse kam: {labels[lo]} ({f(vals[lo])}).",
        ]
        if steps[up] > 0:
            lines.append(f"  Sabse bada uchhaal: {labels[up]} -> {labels[up + 1]} ({f(steps[up], True)} {unit}).")
        if steps[down] < 0:
            lines.append(f"  Sabse badi giravat: {labels[down]} -> {labels[down + 1]} ({f(steps[down], True)} {unit}).")
        lines.append(f"  Haal ka badlav: {labels[-2]} -> {labels[-1]} ({f(steps[-1], True)}{_pct_text(steps[-1], vals[-2])}).")
        if outlier:
            lines.append(f"  Dhyan dein: {labels[hi]} baaki sab se kaafi upar hai; bharosa karne se pehle iska data check karein.")
        return "\n".join(lines)
    lines = [
        header,
        f"  Overall: {_dir_text(direction, lang)} across these {n} {step}.",
        f"  Start -> end: {labels[0]} {f(vals[0])} -> {labels[-1]} {f(vals[-1])} {unit} "
        f"({f(change, True)}{_pct_text(change, vals[0])}).",
        f"  Highest: {labels[hi]} ({f(vals[hi])}) | Lowest: {labels[lo]} ({f(vals[lo])}).",
    ]
    if steps[up] > 0:
        lines.append(f"  Biggest rise: {labels[up]} -> {labels[up + 1]} ({f(steps[up], True)} {unit}).")
    if steps[down] < 0:
        lines.append(f"  Biggest fall: {labels[down]} -> {labels[down + 1]} ({f(steps[down], True)} {unit}).")
    lines.append(f"  Latest move: {labels[-2]} -> {labels[-1]} ({f(steps[-1], True)}{_pct_text(steps[-1], vals[-2])}).")
    if outlier:
        lines.append(f"  Note: {labels[hi]} stands far above the rest; check that data before trusting it.")
    return "\n".join(lines)


def describe_forecast(last_label, last_value: float, labels, values, unit: str, decimals: int,
                      metrics: pd.DataFrame | None, lang: str = "en") -> str:
    """Plain-language reading of a history + forecast graph."""
    vals = np.asarray(list(values), dtype=float)
    labels = [str(label) for label in labels]

    def f(value: float, signed: bool = False) -> str:
        return _fmt(value, lang, decimals, signed)

    first = vals[0] - last_value
    lines = [
        _t(lang, "How to read this graph:", "Graph kaise padhein:"),
        _t(lang, "  Black line = recorded history. Red line = the model's estimate (not a reading).",
           "  Kaali line = recorded history. Laal line = model ka anuman (asli reading nahi)."),
        _t(lang,
           f"  Next: {labels[0]} estimate {f(vals[0])} {unit} "
           f"({f(first, True)}{_pct_text(first, last_value)} vs last recorded {last_label}).",
           f"  Agla: {labels[0]} ka anuman {f(vals[0])} {unit} "
           f"(pichle recorded {last_label} se {f(first, True)}{_pct_text(first, last_value)})."),
    ]
    if len(vals) > 1:
        overall = vals[-1] - last_value
        pct = 100 * overall / abs(last_value) if abs(last_value) > 1e-9 else 0.0
        direction = _dir_text(_direction(pct, 2.0), lang)
        lines.append(_t(lang,
                        f"  By {labels[-1]}: {f(vals[-1])} {unit} ({f(overall, True)}{_pct_text(overall, last_value)}) "
                        f"-> forecast is {direction}.",
                        f"  {labels[-1]} tak: {f(vals[-1])} {unit} ({f(overall, True)}{_pct_text(overall, last_value)}) "
                        f"-> anuman {direction}."))
    else:
        pct = 100 * first / abs(last_value) if abs(last_value) > 1e-9 else 0.0
        direction = _dir_text(_direction(pct, 2.0), lang)
        lines.append(_t(lang, f"  In short: the model expects the next value to be {direction}.",
                        f"  Seedhi baat: model ke hisaab se agla value {direction}."))
    if metrics is not None and not metrics.empty and "r2" in metrics and metrics["r2"].max() < 0:
        lines.append(_t(lang, "  Caution: every model scored a negative R2 in the holdout check, so treat this as rough.",
                        "  Saavdhan: holdout test mein har model ka R2 negative tha, isliye ise sirf rough anuman maanein."))
    return "\n".join(lines)


def describe_model_comparison(metrics: pd.DataFrame, unit: str, decimals: int, lang: str = "en") -> str:
    rmse = metrics.groupby("model")["rmse"].mean().sort_values()
    best = _fmt(rmse.iloc[0], lang, decimals)
    lines = [
        _t(lang, "How to read this graph:", "Graph kaise padhein:"),
        _t(lang, "  Each bar is a model's typical error (RMSE) on held-out data. Shorter bar = more accurate.",
           "  Har bar ek model ki typical galti (RMSE) dikhata hai. Chhota bar = zyada sahi model."),
        _t(lang, f"  Best: {rmse.index[0]} (RMSE {best} {unit}).",
           f"  Sabse accha: {rmse.index[0]} (RMSE {best} {unit})."),
    ]
    if len(rmse) > 1:
        gap = 100 * (rmse.iloc[-1] - rmse.iloc[0]) / rmse.iloc[-1] if rmse.iloc[-1] > 0 else 0.0
        worst = _fmt(rmse.iloc[-1], lang, decimals)
        lines.append(_t(lang, f"  Worst: {rmse.index[-1]} (RMSE {worst} {unit}); the best has {gap:.0f}% less error.",
                        f"  Sabse kamzor: {rmse.index[-1]} (RMSE {worst} {unit}); "
                        f"sabse acche model ki galti {gap:.0f}% kam hai."))
    return "\n".join(lines)


def explain_graph_followup(t: str, ctx: ChatContext) -> str | None:
    """'explain the graph' / 'trend samjhao' after a graph: re-read that graph instead of redrawing it."""
    if ctx.last_graph_describe is None or re.search(r"\d", t):
        return None
    if not (wants_explanation(t) or wants_trend(t)):
        return None
    if ctx.last_graph_kind == "graph_metrics" and not wants_explanation(t):
        return None  # a bar chart of model errors has no trend; read the history instead
    if not (wants_graph(t) or wants_trend(t) or (ctx.last_topic or "").startswith("graph")):
        return None
    if any(w in t for w in ("comparison", "accuracy")):
        requested = "graph_metrics"
    elif any(w in t for w in ("forecast", "predict")):
        requested = "graph_forecast"
    else:
        requested = None
    if requested is not None and requested != ctx.last_graph_kind:
        return None
    summary = ctx.last_graph_describe(ctx.lang)
    return (summary.replace("How to read this graph:", "How to read the last graph:", 1)
            .replace("Graph kaise padhein:", "Pichla graph kaise padhein:", 1))


GraphResult = tuple[Path, Callable[[str], str]]


def graph_reply(ctx: ChatContext, kind: str, result: GraphResult, open_file: bool) -> tuple[str, ChatContext]:
    path, describe = result
    ctx.last_topic = ctx.last_graph_kind = kind
    ctx.last_graph_describe = describe
    return f"{_open_png(path, open_file, ctx.lang)}\n\n{describe(ctx.lang)}", ctx


def _short_value(value: float, unit: str) -> str:
    """Compact value for chart labels: QTY in lakh, everything else with one decimal."""
    if unit == "QTY" and abs(value) >= 1e5:
        return f"{value / 1e5:.2f} lakh"
    if unit == "QTY":
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def _label_box(ax, x, y, text: str, color: str, below: bool = False) -> None:
    ax.annotate(text, (x, y), xytext=(0, -16 if below else 12), textcoords="offset points",
                ha="center", va="top" if below else "bottom", fontsize=8.5, color=color, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "lw": 0.8, "alpha": 0.9})


def mark_high_low(ax, xs, ys, unit: str) -> None:
    """Mark the highest and lowest recorded points directly on the chart."""
    ys = np.asarray(list(ys), dtype=float)
    xs = list(xs)
    if len(ys) < 2 or ys.max() == ys.min():
        return
    hi, lo = int(ys.argmax()), int(ys.argmin())
    ax.scatter([xs[hi]], [ys[hi]], marker="^", s=110, color="#2ca02c", zorder=5, label="Highest")
    ax.scatter([xs[lo]], [ys[lo]], marker="v", s=110, color="#ff7f0e", zorder=5, label="Lowest")
    _label_box(ax, xs[hi], ys[hi], f"High: {_short_value(ys[hi], unit)}", "#2ca02c")
    _label_box(ax, xs[lo], ys[lo], f"Low: {_short_value(ys[lo], unit)}", "#ff7f0e", below=True)
    ax.margins(y=0.22)  # room for the labels above and below the line


def label_forecast_points(ax, xs, ys, unit: str, max_labels: int = 8) -> None:
    """Write each forecast value next to its point; with more than max_labels points, only the last one."""
    xs, ys = list(xs), list(ys)
    picks = range(len(ys)) if len(ys) <= max_labels else [len(ys) - 1]
    for i in picks:
        _label_box(ax, xs[i], ys[i], _short_value(float(ys[i]), unit), "#d62728")
    ax.margins(y=0.22)


def lakh_axis(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e5:.1f}"))


def draw_rmse_bars(metrics: pd.DataFrame, out: Path, unit: str) -> None:
    rmse = metrics.groupby("model")["rmse"].mean().sort_values()
    kinds = metrics.groupby("model")["type"].first() if "type" in metrics else pd.Series(dtype=str)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # best in green, simple baselines grey, neural models blue
    colors = ["#2ca02c" if i == 0 else ("#7f7f7f" if kinds.get(name) == "baseline" else "#1f77b4")
              for i, name in enumerate(rmse.index)]
    bars = ax.bar(rmse.index, rmse.values, color=colors)
    ax.bar_label(bars, labels=[_short_value(v, unit) for v in rmse.values], padding=3, fontsize=9)
    ax.set(title="Test RMSE by model (shorter = better; green = best, grey = simple baseline)", xlabel="Model",
           ylabel=f"RMSE ({'QTY' if unit == 'QTY' else 'm³'})")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.margins(y=0.15)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_recent_history(history: pd.DataFrame, n: int, run_dir: Path) -> GraphResult:
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
    mark_high_low(ax, tail["date"], tail["supply_m3"], "m3")
    ax.set(title=f"Daily water supply — last {n} days", ylabel="Supply (m³)", xlabel="Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    values, labels = tail["supply_m3"].to_list(), tail["date"].dt.date.to_list()
    return out, lambda lang: describe_trend(values, labels, "m3", "days", 1, lang)


def plot_forecast(model, scaler, meta, history, days: int, run_dir: Path,
                  metrics: pd.DataFrame | None = None) -> GraphResult:
    seq_len = int(meta.get("sequence_length", 30))
    context_n = min(60, len(history))
    hist_tail = history.tail(context_n).copy()
    hist_tail["date"] = pd.to_datetime(hist_tail["date"])
    recent = history["supply_m3"].iloc[-seq_len:]
    fc = forecast_future(model, scaler, recent, days, seq_len)
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
    mark_high_low(ax, hist_tail["date"], hist_tail["supply_m3"], "m3")
    label_forecast_points(ax, fut_dates, fc, "m3", max_labels=1)  # daily points sit too close for more labels
    ax.set(title="Water supply: recent history + model forecast", ylabel="Supply (m³)", xlabel="Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    last_value, labels = float(recent.iloc[-1]), [d.date() for d in fut_dates]
    return out, lambda lang: describe_forecast(last_date.date(), last_value, labels, fc, "m3", 1, metrics, lang)


def plot_model_comparison(metrics: pd.DataFrame, run_dir: Path, unit: str = "m3") -> GraphResult:
    out_dir = run_dir / "chatbot_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model_comparison.png"
    draw_rmse_bars(metrics, out, unit)
    decimals = 2 if unit == "QTY" else 1
    return out, lambda lang: describe_model_comparison(metrics, unit, decimals, lang)


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
    return forecast_future(model, scaler, history["supply_m3"], count, seq_len)


def do_period_forecast(model, scaler, meta, history: pd.DataFrame, count: int, lang: str = "en") -> str:
    count = max(1, min(12, count))
    forecasts = forecast_period_values(model, scaler, meta, history, count)
    periods = next_periods(history, count)
    last_label, last_value = history["billing_period"].iloc[-1], float(history["supply_m3"].iloc[-1])
    if lang == "hi":
        if count == 1:
            (start, end, label), value = periods[0], float(forecasts[0])
            change = value - last_value
            return "\n".join([
                f"Agle bill ({label}, {start.date()} se {end.date()}) mein lagbhag {_fmt(value, lang, 2)} QTY ka anuman hai.",
                f"Pichla recorded bill {last_label} = {_fmt(last_value, lang, 2)} QTY tha, "
                f"yaani {_fmt(change, lang, 2, True)}{_pct_text(change, last_value)} ka badlav.",
                "Yeh model ka anuman hai, asli reading nahi. QTY ki unit abhi confirm nahi hai.",
            ])
        lines = [f"Pichla recorded bill: {last_label} = {_fmt(last_value, lang, 2)} QTY."]
        for (start, end, label), value in zip(periods, forecasts):
            lines.append(f"  {label} ({start.date()} se {end.date()}): lagbhag {_fmt(float(value), lang, 2)} QTY ka anuman.")
        lines.append("Yeh model ke anuman hain (har bill do mahine ka); QTY ki unit abhi confirm nahi hai.")
        return "\n".join(lines)
    lines = [f"Last recorded period: {last_label} = {last_value:,.2f} QTY."]
    for (start, end, label), value in zip(periods, forecasts):
        lines.append(f"  Forecast {label} ({start.date()} to {end.date()}): {value:,.2f} QTY.")
    lines.append("This is a model estimate for a two-month billing period; QTY's source unit is unconfirmed.")
    return "\n".join(lines)


def explain_period_forecast(model, scaler, meta, history: pd.DataFrame, metrics: pd.DataFrame | None,
                            lang: str = "en") -> str:
    seq_len = int(meta.get("sequence_length", 6))
    forecast = forecast_period_values(model, scaler, meta, history, 1)[0]
    next_label = next_periods(history, 1)[0][2]
    value = _fmt(float(forecast), lang, 2)
    variant = meta.get("variant", "cnn_lstm_attention")
    lines = [
        _t(lang, "This is a model estimate, not a meter reading.", "Yeh model ka anuman hai, meter reading nahi."),
        _t(lang, f"The {variant} model looks at the last {seq_len} billing periods and estimates the next period.",
           f"{variant} model pichle {seq_len} bills dekh kar agle bill ka anuman lagata hai."),
        _t(lang, f"Next period ({next_label}) estimate: {value} QTY.",
           f"Agle bill ({next_label}) ka anuman: lagbhag {value} QTY."),
        _t(lang, "Each period covers about two months. The source unit for QTY has not been confirmed.",
           "Har bill lagbhag do mahine ka hota hai. QTY ki unit abhi confirm nahi hai."),
    ]
    if metrics is not None and not metrics.empty:
        lines.append(accuracy_text(metrics, meta, "QTY", 2, lang))
    return "\n".join(lines)


def explain_period_model(meta: dict, run_dir: Path, metrics: pd.DataFrame | None, lang: str = "en") -> str:
    seq_len = int(meta.get("sequence_length", 6))
    variant = meta.get("variant", "cnn_lstm_attention")
    lines = [
        _t(lang, "This model is trained on Kharghar's combined customer QTY per two-month billing period.",
           "Yeh model Kharghar ke sabhi customers ki do-mahine ki total QTY par train hua hai."),
        _t(lang, f"It uses the last {seq_len} periods to estimate the next period.",
           f"Yeh pichle {seq_len} bills dekh kar agle bill ka anuman lagata hai."),
        _t(lang, f"The saved network is {variant} (file {run_dir.name}/{weights_file(run_dir).name}).",
           f"Save kiya hua network {variant} hai (file {run_dir.name}/{weights_file(run_dir).name})."),
        _t(lang, "It was chosen as the neural variant with the lowest walk-forward error; one time step is one billing period.",
           "Yeh walk-forward test mein sabse kam galti wala neural model tha; ek step matlab ek bill (do mahine)."),
    ]
    if metrics is not None and not metrics.empty:
        lines.append(accuracy_text(metrics, meta, "QTY", 2, lang))
    lines.append(_t(lang, "There are few periods, so treat forecasts as exploratory.",
                    "Data mein bills kam hain, isliye anuman ko sirf andaaza maanein."))
    return "\n".join(lines)


def plot_period_history(history: pd.DataFrame, count: int, run_dir: Path) -> GraphResult:
    count = max(1, min(len(history), count))
    tail = history.tail(count)
    out_dir = run_dir / "chatbot_graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"history_last{count}_periods.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(tail["date"], tail["supply_m3"], marker="o", linewidth=1.8, color="#1f77b4")
    mark_high_low(axis, tail["date"], tail["supply_m3"], "QTY")
    lakh_axis(axis)
    axis.set(title=f"Kharghar total QTY — last {count} billing periods",
             ylabel="Total QTY in lakh (source unit unconfirmed)", xlabel="Billing period")
    axis.set_xticks(tail["date"], tail["billing_period"], rotation=35, ha="right")
    if count >= 2:
        axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    values, labels = tail["supply_m3"].to_list(), tail["billing_period"].to_list()
    return out, lambda lang: describe_trend(values, labels, "QTY", "billing periods", 2, lang)


def plot_period_forecast(model, scaler, meta, history: pd.DataFrame, count: int,
                         run_dir: Path, metrics: pd.DataFrame | None = None) -> GraphResult:
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
    mark_high_low(axis, x[:len(tail)], values[:len(tail)], "QTY")
    label_forecast_points(axis, x[len(tail):], values[len(tail):], "QTY")
    lakh_axis(axis)
    axis.set(title="Kharghar total QTY by billing period",
             ylabel="Total QTY in lakh (source unit unconfirmed)", xlabel="Billing period")
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    last_label, last_value = history["billing_period"].iloc[-1], float(history["supply_m3"].iloc[-1])
    future_labels = [item[2] for item in future]
    return out, lambda lang: describe_forecast(last_label, last_value, future_labels, forecasts, "QTY", 2,
                                               metrics, lang)


MONTH_NUMBERS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "JLY": 7,
                 "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
MONTH_PATTERN = (r"JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUNE?|JULY?|JLY|AUG(?:UST)?"
                 r"|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?")


def _period_value(target_start: pd.Timestamp, model, scaler, meta,
                  history: pd.DataFrame, lang: str = "en") -> tuple[str, float, str] | str:
    """Recorded or forecast total for the billing period starting on target_start, or an error message."""
    recorded = history[pd.to_datetime(history["date"]).dt.normalize() == target_start]
    if not recorded.empty:
        row = recorded.iloc[0]
        return row["billing_period"], float(row["supply_m3"]), "recorded"
    next_start = pd.to_datetime(history["period_end"].iloc[-1]) + pd.Timedelta(days=1)
    month_offset = (target_start.year - next_start.year) * 12 + target_start.month - next_start.month
    if month_offset < 0:
        when = target_start.strftime("%b %Y")
        return _t(lang, f"There is no recorded total for the period starting {when} in this run's history.",
                  f"{when} se shuru hone wale bill ka koi recorded total is run ki history mein nahi hai.")
    steps = month_offset // 2 + 1
    if steps > 12:
        return _t(lang, "That period is more than 12 cycles ahead; this chatbot only forecasts up to 12 periods.",
                  "Yeh bill 12 cycles se zyada aage hai; yeh chatbot sirf 12 bills tak ka anuman lagata hai.")
    forecasts = forecast_period_values(model, scaler, meta, history, steps)
    return next_periods(history, steps)[-1][2], float(forecasts[-1]), "forecast"


def answer_named_period(text: str, model, scaler, meta, history: pd.DataFrame, lang: str = "en") -> str | None:
    match = re.search(rf"\b({MONTH_PATTERN})\s*(?:-|/|\s|\bto\b)\s*({MONTH_PATTERN})\s*(20\d{{2}})?\b", text.upper())
    if not match:
        return None
    start_month = MONTH_NUMBERS[match.group(1)[:3]]
    end_month = MONTH_NUMBERS[match.group(2)[:3]]
    if end_month != start_month % 12 + 1:
        pair = f"{match.group(1)}-{match.group(2)}"
        return _t(lang, f"{pair} is not two back-to-back months. Ask for a pair like OCT-NOV2026 or 'next billing cycle'.",
                  f"{pair} lagataar do mahine nahi hain. OCT-NOV2026 jaisa pair ya 'agla bill' poochiye.")
    label_months = f"{match.group(1)[:3]}-{match.group(2)[:3]}"
    next_start = pd.to_datetime(history["period_end"].iloc[-1]) + pd.Timedelta(days=1)
    if match.group(3):
        end_year = int(match.group(3))  # labels carry the end month's year, e.g. DEC-JAN2024
        start_year = end_year - 1 if start_month == 12 else end_year
        assumed_year = ""
    else:
        # No year given: use the first such pair that has not been recorded yet.
        start_year = next_start.year
        if pd.Timestamp(start_year, start_month, 1) < next_start - pd.DateOffset(months=1):
            start_year += 1
        end_year = start_year + 1 if start_month == 12 else start_year
        older = f"{label_months}{end_year - 1}"
        assumed_year = _t(lang, f" (I assumed {end_year}; add a year, e.g. {older}, for an older one)",
                          f" (maine {end_year} maana hai; purana chahiye to saal likhiye, jaise {older})")
    target_start = pd.Timestamp(start_year, start_month, 1)
    requested_label = f"{label_months}{end_year}"

    month_offset = (target_start.year - next_start.year) * 12 + target_start.month - next_start.month
    if month_offset % 2 == 0:
        result = _period_value(target_start, model, scaler, meta, history, lang)
        if isinstance(result, str):
            return result
        label, value, kind = result
        if kind == "recorded":
            return _t(lang, f"Recorded total for {label}: {value:,.2f} QTY{assumed_year}.",
                      f"{label} ka recorded total: {_fmt(value, lang, 2)} QTY{assumed_year}.")
        steps = month_offset // 2 + 1
        return (_t(lang, f"Estimate through {requested_label}{assumed_year}:\n",
                   f"{requested_label} tak ka anuman{assumed_year}:\n")
                + do_period_forecast(model, scaler, meta, history, steps, lang))

    # The pair straddles two bills (e.g. NOV-DEC sits across OCT-NOV and DEC-JAN): estimate it from both halves.
    first = _period_value(target_start - pd.DateOffset(months=1), model, scaler, meta, history, lang)
    second = _period_value(target_start + pd.DateOffset(months=1), model, scaler, meta, history, lang)
    for part in (first, second):
        if isinstance(part, str):
            return part
    (label_a, value_a, kind_a), (label_b, value_b, kind_b) = first, second
    estimate = value_a / 2 + value_b / 2
    if lang == "hi":
        kind_hi = {"recorded": "recorded", "forecast": "anuman"}
        return "\n".join([
            f"{requested_label} Kharghar ka koi billing period nahi hai{assumed_year}.",
            f"Bills {label_a[:7]} aur {label_b[:7]} ke hote hain, isliye yeh do mahine do alag bills mein aate hain:",
            f"  {label_a} ({kind_hi[kind_a]}): {_fmt(value_a, lang, 2)} QTY",
            f"  {label_b} ({kind_hi[kind_b]}): {_fmt(value_b, lang, 2)} QTY",
            f"{requested_label} ka rough anuman: lagbhag {_fmt(estimate, lang, 2)} QTY "
            "(dono bills ka aadha, yeh maan kar ki har bill ka paani dono mahino mein barabar bata hai).",
        ])
    return "\n".join([
        f"{requested_label} is not one of Kharghar's billing periods{assumed_year}.",
        f"Bills cover {label_a[:7]} and {label_b[:7]}, so these two months fall in two different bills:",
        f"  {label_a} ({kind_a}): {value_a:,.2f} QTY",
        f"  {label_b} ({kind_b}): {value_b:,.2f} QTY",
        f"Rough {requested_label} estimate: about {estimate:,.2f} QTY "
        "(half of each bill, assuming use is spread evenly over each bill's two months).",
    ])


def answer_period(text: str, model, scaler, meta, history, metrics, run_dir, ctx,
                  *, open_graph: bool = True) -> tuple[str, ChatContext]:
    t = text.strip().lower()
    lang = ctx.lang
    if t in {"exit", "quit", "q"}:
        return "EXIT", ctx
    if t in {"help", "?", "madad"}:
        return _t(lang, PERIOD_HELP, PERIOD_HELP_HI), ctx
    followup = explain_graph_followup(t, ctx)
    if followup is not None:
        return followup, ctx
    if wants_trend(t) and not wants_graph(t):
        nums = [int(value) for value in re.findall(r"\d+", t)]
        tail = history.tail(max(1, min(len(history), nums[0] if nums else 6)))
        return describe_trend(tail["supply_m3"], tail["billing_period"], "QTY", "billing periods", 2, lang), ctx
    if wants_graph(t):
        nums = [int(value) for value in re.findall(r"\d+", t)]
        if "comparison" in t or "metric" in t or "model" in t or "accuracy" in t:
            if metrics is None:
                return _t(lang, "No metrics.csv in this run.", "Is run mein metrics.csv nahi hai."), ctx
            return graph_reply(ctx, "graph_metrics", plot_model_comparison(metrics, run_dir, "QTY"), open_graph)
        if any(w in t for w in ("forecast", "predict", "next", "agla", "agle", "anuman")):
            return graph_reply(ctx, "graph_forecast", plot_period_forecast(
                model, scaler, meta, history, nums[0] if nums else 1, run_dir, metrics), open_graph)
        return graph_reply(ctx, "graph_recent", plot_period_history(history, nums[0] if nums else 6, run_dir),
                           open_graph)
    if "dataset" in t or "what data" in t or "which data" in t:
        return _t(lang, PERIOD_DATASET_GUIDE, PERIOD_DATASET_GUIDE_HI), ctx
    if any(phrase in t for phrase in ("how does the model", "how do you predict", "how does this work",
                                      "what is the model", "kaise kaam", "kaise predict", "model kya hai")):
        return explain_period_model(meta, run_dir, metrics, lang), ctx
    if "accur" in t or "metric" in t or "r2" in t or "rmse" in t or "how good" in t or "kitna sahi" in t:
        if metrics is None or metrics.empty:
            return _t(lang, "No metrics.csv in this run.", "Is run mein metrics.csv nahi hai."), ctx
        ctx.last_topic = "metrics"
        return accuracy_text(metrics, meta, "QTY", 2, lang), ctx
    if wants_explanation(t):
        if any(w in t for w in ("forecast", "next", "cycle", "agla", "agle", "anuman")) or ctx.last_topic == "forecast":
            return explain_period_forecast(model, scaler, meta, history, metrics, lang), ctx
        if ctx.last_topic == "average":
            return _t(lang, "That is the average total QTY per two-month billing period, not per day or per month.",
                      "Yeh ek do-mahine ke bill ka average total QTY hai, roz ya mahine ka nahi."), ctx
        if ctx.last_topic in {"peak", "lowest"}:
            return _t(lang, "That is the highest or lowest recorded period total, not a forecast.",
                      "Yeh sabse zyada ya sabse kam recorded bill ka total hai, anuman nahi."), ctx
        return _t(lang, "Ask about the next billing cycle, last periods, an average, highest/lowest period, or accuracy; "
                        "then say 'explain this'.",
                  "Pehle kuch poochiye (agla bill, pichle bills, average, sabse zyada/kam, accuracy); "
                  "phir 'samjhao' likhiye."), ctx
    named_period = answer_named_period(text, model, scaler, meta, history, lang)
    if named_period is not None:
        is_forecast = any(marker in named_period for marker in ("Estimate through", "(forecast)", "tak ka anuman", "(anuman)"))
        ctx.last_topic = "forecast" if is_forecast else "recent"
        return named_period, ctx
    if "next month" in t or "tomorrow" in t or "agla mahina" in t or "agle mahine" in t:
        return _t(lang, "The Kharghar data is grouped into two-month billing periods, so it cannot forecast one month "
                        "by itself. Ask 'next billing cycle'.",
                  "Kharghar ka data do-do mahine ke bills mein hai, isliye sirf ek mahine ka anuman nahi ho sakta. "
                  "'agla bill' poochiye."), ctx
    forecast_request = any(word in t for word in ("forecast", "predict", "anuman")) or (
        any(word in t for word in ("next", "agla", "agle", "aane wala", "aane wale"))
        and any(word in t for word in ("period", "cycle", "bill"))
    )
    if forecast_request:
        numbers = [int(value) for value in re.findall(r"\d+", t)]
        count = numbers[0] if numbers else 1
        ctx.last_topic = "forecast"
        return do_period_forecast(model, scaler, meta, history, count, lang), ctx
    if "average" in t or "mean" in t or "ausat" in t:
        numbers = [int(value) for value in re.findall(r"\d+", t)]
        count = max(1, min(len(history), numbers[0] if numbers else 6))
        avg = _fmt(float(history.tail(count)["supply_m3"].mean()), lang, 2)
        ctx.last_topic = "average"
        return _t(lang, f"Average total for the last {count} billing periods: {avg} QTY.",
                  f"Pichle {count} bills ka average total: {avg} QTY."), ctx
    if any(word in t for word in ("highest", "peak", "maximum", "max", "sabse zyada", "sabse jyada")):
        row = history.loc[history["supply_m3"].idxmax()]
        ctx.last_topic = "peak"
        value = _fmt(float(row["supply_m3"]), lang, 2)
        return _t(lang, f"Highest recorded period total: {row['billing_period']} = {value} QTY.",
                  f"Sabse zyada recorded bill: {row['billing_period']} = {value} QTY."), ctx
    if any(word in t for word in ("lowest", "minimum", "min", "least", "sabse kam")):
        row = history.loc[history["supply_m3"].idxmin()]
        ctx.last_topic = "lowest"
        value = _fmt(float(row["supply_m3"]), lang, 2)
        return _t(lang, f"Lowest recorded period total: {row['billing_period']} = {value} QTY.",
                  f"Sabse kam recorded bill: {row['billing_period']} = {value} QTY."), ctx
    if any(word in t for word in ("last", "recent", "pichla", "pichle")):
        numbers = [int(value) for value in re.findall(r"\d+", t)]
        count = max(1, min(len(history), numbers[0] if numbers else 1))
        selected = history.tail(count)
        ctx.last_topic, ctx.last_n = "recent", count
        lines = [_t(lang, "Recent billing-period totals:", "Haal ke bills ke total:")]
        lines.extend(f"  {row.billing_period}: {_fmt(float(row.supply_m3), lang, 2)} QTY" for row in selected.itertuples())
        return "\n".join(lines), ctx
    if "hello" in t or t == "hi" or "namaste" in t:
        return _t(lang, "Hi. Try 'next billing cycle', 'average last 6 periods', 'highest period', or 'help'.",
                  "Namaste! 'agla bill', 'pichle 6 bills ka average', 'sabse zyada bill' ya 'help' poochiye."), ctx
    return _t(lang, "I answer from the Kharghar billing-period model and history. Try 'next billing cycle', "
                    "'last 4 periods', 'average last 6 periods', 'accuracy', or 'help'.",
              "Main Kharghar billing-period model aur history se jawab deta hoon. 'agla bill', 'pichle 4 bills', "
              "'pichle 6 bills ka average', 'accuracy' ya 'help' try kijiye."), ctx


def handle_graph_request(
    t: str, ctx: ChatContext, model, scaler, meta, history, metrics, run_dir: Path, open_file: bool
) -> tuple[str, ChatContext] | None:
    if not wants_graph(t):
        return None
    nums = re.findall(r"\d+", t)
    if "comparison" in t or "metric" in t or "accuracy" in t or "model" in t:
        if metrics is None:
            return "No metrics.csv in this run — train with run_experiment.py first.", ctx
        return graph_reply(ctx, "graph_metrics", plot_model_comparison(metrics, run_dir), open_file)
    if "forecast" in t or "tomorrow" in t or "predict" in t:
        days = max(1, min(60, int(nums[0]))) if nums else ctx.last_forecast_days
        ctx.last_forecast_days = days
        return graph_reply(ctx, "graph_forecast",
                           plot_forecast(model, scaler, meta, history, days, run_dir, metrics), open_file)
    if t.strip() in {"graph", "chart", "plot", "graph this", "plot this", "show graph"}:
        if ctx.last_topic == "forecast":
            return graph_reply(ctx, "graph_forecast", plot_forecast(
                model, scaler, meta, history, ctx.last_forecast_days, run_dir, metrics), open_file)
        if ctx.last_topic == "recent":
            return graph_reply(ctx, "graph_recent", plot_recent_history(history, ctx.last_n, run_dir), open_file)
        return (
            "Say what to plot, e.g. 'graph last 30 days' or 'plot forecast 7 days' or 'graph model comparison'.",
            ctx,
        )
    n = int(nums[0]) if nums else ctx.last_n
    if "last" in t or "recent" in t or "history" in t or "day" in t:
        n = max(1, min(365, n))
    else:
        n = max(7, min(365, n if nums else 30))
    ctx.last_n = n
    return graph_reply(ctx, "graph_recent", plot_recent_history(history, n, run_dir), open_file)


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
            "samjha",
            "matlab",
            "kya dikha",
        )
    )


def do_forecast(model, scaler, meta, history, days: int) -> str:
    seq_len = int(meta.get("sequence_length", 30))
    recent = history["supply_m3"].iloc[-seq_len:]
    fc = forecast_future(model, scaler, recent, days, seq_len)
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
    t = text.strip().lower()
    switched = set_language(t, ctx)
    if switched is not None:
        return switched, ctx
    ctx.lang = detect_lang(t, ctx.lang_mode)
    if meta.get("data_frequency") == "two_month_billing_period":
        return answer_period(text, model, scaler, meta, history, metrics, run_dir, ctx, open_graph=open_graph)
    # Daily mode: graph readings follow the reply language; other daily answers are English only.
    if t in {"exit", "quit", "q"}:
        return "EXIT", ctx
    if t in {"help", "?"}:
        return HELP, ctx
    followup = explain_graph_followup(t, ctx)
    if followup is not None:
        return followup, ctx
    if wants_trend(t) and not wants_graph(t):
        nums = re.findall(r"\d+", t)
        tail = history.tail(max(2, min(len(history), int(nums[0]) if nums else 30)))
        return describe_trend(tail["supply_m3"], pd.to_datetime(tail["date"]).dt.date, "m3", "days", 1,
                              ctx.lang), ctx
    graph = handle_graph_request(t, ctx, model, scaler, meta, history, metrics, run_dir, open_graph)
    if graph is not None:
        return graph
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
        ctx.last_topic = "metrics"
        return accuracy_text(metrics, meta, "m3", 1, ctx.lang), ctx
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
    # Replace characters the console cannot show (e.g. ³, ²) instead of raising UnicodeEncodeError.
    sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    run_dir, model, scaler, meta, history, metrics = load_all(args)
    if meta.get("data_frequency") == "two_month_billing_period":
        print(f"\nKharghar billing-period chatbot ready (offline). Run: {run_dir.name} | periods: {len(history)}")
        print("Ask about the next billing cycle or type 'help'. Type 'exit' to quit.")
        print("Hinglish mein bhi pooch sakte hain (jaise 'agla bill kitna hoga?'), ya 'hinglish' likhiye.\n")
    else:
        print(f"\nWater-supply chatbot ready (offline). Run: {run_dir.name} | history rows: {len(history)}")
        print("Ask about consumption or type 'help'. After any answer, try 'explain this'. Type 'exit' to quit.\n")
    ctx = ChatContext(lang_mode={"auto": "auto", "en": "en", "hinglish": "hi"}[args.lang])
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nchatbot: bye.")
            break
        if not text:
            continue
        try:
            reply, ctx = answer(
                text, model, scaler, meta, history, metrics, run_dir, ctx,
                open_graph=not args.no_open_graph,
            )
        except Exception as exc:  # one bad question must not end the chat session
            print(f"chatbot:\nSorry, that question failed ({type(exc).__name__}: {exc}).\n"
                  "The chat is still running; try rephrasing or type 'help'.\n")
            continue
        if reply == "EXIT":
            print("chatbot: bye.")
            break
        print("chatbot:\n" + reply + "\n")


if __name__ == "__main__":
    main()
