"""FastAPI backend for the Angular ChatGPT-style water-forecast chatbot.

Wraps the trained CNN-LSTM-Attention bundles (daily + Kharghar billing-period)
and the rule-based logic in chatbot.py, but returns JSON with structured
chart data so the Angular UI can render interactive charts (no PNG files).

Run (PowerShell, from project root):
  uvicorn api.main:app --reload --port 8000

Endpoints:
  GET  /api/health
  GET  /api/runs                 -> available trained runs (daily + billing)
  GET  /api/runs/{run_id}        -> run detail: meta, metrics, history summary
  GET  /api/history?run_id=&count= -> history points for charts/tables
  GET  /api/forecast?run_id=&count= -> model forecast points
  GET  /api/metrics?run_id=       -> walk-forward metrics table
  POST /api/chat {message, run_id?, session_id?, lang?} -> {reply, chart?, data?}
"""
from __future__ import annotations

import re
import sys
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import chatbot as bot  # noqa: E402
from water_forecast.inference import (  # noqa: E402
    find_latest_run,
    forecast_future,
    load_bundle,
    run_frequency,
)

OUTPUTS = ROOT / "outputs"

app = FastAPI(title="Urban Water Forecast Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------- cache
_RUN_CACHE: dict[str, dict] = {}
_SESSIONS: dict[str, bot.ChatContext] = {}
_SESSION_RUN: dict[str, str] = {}


def _available_runs() -> list[Path]:
    if not OUTPUTS.exists():
        return []
    return sorted(
        [p for p in OUTPUTS.iterdir() if p.is_dir() and (p / "model_bundle.json").exists()],
        key=lambda p: p.name,
    )


def _run_summary(run_dir: Path) -> dict:
    import json

    meta = json.loads((run_dir / "model_bundle.json").read_text())
    freq = meta.get("data_frequency", "daily")
    hist_file = run_dir / ("cleaned_period_totals.csv" if freq == "two_month_billing_period" else "cleaned_daily_supply.csv")
    n_history = None
    last_label: str | None = None
    last_value: float | None = None
    if hist_file.exists():
        try:
            df = pd.read_csv(hist_file, usecols=["billing_period", "total_qty"]) if freq == "two_month_billing_period" else pd.read_csv(hist_file, usecols=["date", "supply_m3"])
            n_history = len(df)
            if freq == "two_month_billing_period":
                last_label = str(df["billing_period"].iloc[-1])
                last_value = float(df["total_qty"].iloc[-1])
            else:
                last_label = str(df["date"].iloc[-1])
                last_value = float(df["supply_m3"].iloc[-1])
        except Exception:
            pass
    return {
        "id": run_dir.name,
        "kind": "billing" if freq == "two_month_billing_period" else "daily",
        "frequency": freq,
        "variant": meta.get("variant"),
        "variant_label": _model_name(meta.get("variant") or ""),
        "sequence_length": int(meta.get("sequence_length", 0)),
        "history_rows": n_history,
        "last_label": last_label,
        "last_value": last_value,
        "evaluation": meta.get("evaluation", {}),
    }


def _load_run(run_id: str | None) -> tuple[str, dict]:
    """Load (and cache) model bundle + history + metrics for a run."""
    if run_id is None:
        latest = find_latest_run(OUTPUTS)
        if latest is None:
            raise HTTPException(500, "No trained runs found in outputs/. Train once with run_experiment.py.")
        run_id = latest.name
    run_dir = OUTPUTS / run_id
    if not run_dir.is_dir() or not (run_dir / "model_bundle.json").exists():
        valid = [p.name for p in _available_runs()]
        raise HTTPException(404, f"Unknown run '{run_id}'. Available: {valid}")
    if run_id not in _RUN_CACHE:
        model, scaler, meta = load_bundle(run_dir)
        freq = meta.get("data_frequency", "daily")
        if freq == "two_month_billing_period":
            hist_path = run_dir / "cleaned_period_totals.csv"
            if not hist_path.exists():
                raise HTTPException(500, f"Run {run_id} has no cleaned_period_totals.csv")
            history = bot.load_period_history(hist_path)
        else:
            history = pd.read_csv(run_dir / "cleaned_daily_supply.csv", parse_dates=["date"])
        metrics = None
        if (run_dir / "metrics.csv").exists():
            metrics = bot.active_metrics(pd.read_csv(run_dir / "metrics.csv"))
        bot.attach_forecast_errors(meta, run_dir, metrics)
        _RUN_CACHE[run_id] = {
            "dir": run_dir, "model": model, "scaler": scaler,
            "meta": meta, "history": history, "metrics": metrics,
        }
    return run_id, _RUN_CACHE[run_id]


def _history_points(history: pd.DataFrame, count: int, billing: bool) -> dict:
    tail = history.tail(max(1, min(len(history), count)))
    if billing:
        labels = tail["billing_period"].astype(str).tolist()
    else:
        labels = pd.to_datetime(tail["date"]).dt.strftime("%Y-%m-%d").tolist()
    values = [float(v) for v in tail["supply_m3"].tolist()]
    return {"labels": labels, "values": values}


def _forecast_points(bundle: dict, count: int) -> dict:
    meta, history = bundle["meta"], bundle["history"]
    billing = meta.get("data_frequency") == "two_month_billing_period"
    count = max(1, min(12 if billing else 60, count))
    if billing:
        fc = bot.forecast_period_values(bundle["model"], bundle["scaler"], meta, history, count)
        periods = bot.next_periods(history, count)
        labels = [p[2] for p in periods]
        last_label = str(history["billing_period"].iloc[-1])
    else:
        seq_len = int(meta.get("sequence_length", 30))
        recent = history["supply_m3"].iloc[-seq_len:]
        fc = forecast_future(bundle["model"], bundle["scaler"], recent, count, seq_len)
        last_date = pd.to_datetime(history["date"].iloc[-1])
        labels = [(last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, count + 1)]
        last_label = last_date.strftime("%Y-%m-%d")
    values = [float(v) for v in np.asarray(fc, dtype=float).tolist()]
    # likely range per step: estimate ± the model's average walk-forward error that far ahead
    errors = [bot.forecast_error(meta, step) for step in range(1, len(values) + 1)]
    has_range = all(e is not None for e in errors)
    return {
        "labels": labels, "values": values,
        "lower": [v - e for v, e in zip(values, errors)] if has_range else None,
        "upper": [v + e for v, e in zip(values, errors)] if has_range else None,
        "last_label": last_label, "last_value": float(history["supply_m3"].iloc[-1]),
        "unit": "QTY" if billing else "m3",
    }


def _metrics_rows(bundle: dict) -> list[dict]:
    m = bundle["metrics"]
    if m is None or m.empty:
        return []
    out = []
    for r in m.to_dict(orient="records"):
        out.append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()})
    return out


def _build_chart(bundle: dict, ctx: bot.ChatContext) -> dict | None:
    """Translate the chatbot's last graph into Chart.js-ready JSON."""
    kind = ctx.last_graph_kind
    if not kind:
        return None
    meta, history, metrics = bundle["meta"], bundle["history"], bundle["metrics"]
    billing = meta.get("data_frequency") == "two_month_billing_period"
    unit = "QTY" if billing else "m3"
    try:
        if kind == "graph_metrics":
            if metrics is None or metrics.empty:
                return None
            rmse = metrics.groupby("model")["rmse"].mean().sort_values()
            return {
                "kind": "bar", "title": "Test RMSE by model (shorter = better)",
                "x_label": "Model", "y_label": f"RMSE ({unit})",
                "labels": [_model_name(m) for m in rmse.index],
                "datasets": [{"label": "RMSE", "data": [float(v) for v in rmse.values]}],
            }
        if kind == "graph_forecast":
            n = ctx.last_forecast_days if not billing else max(1, min(6, ctx.last_forecast_days))
            fc = _forecast_points(bundle, n)
            hist = _history_points(history, min(8 if billing else 30, len(history)), billing)
            pad = [None] * (len(hist["values"]) - 1)
            datasets = [
                {"label": "Recorded", "data": hist["values"] + [None] * len(fc["labels"])},
                {"label": "Forecast", "data": pad + [hist["values"][-1]] + fc["values"]},
            ]
            if fc["lower"] is not None:
                # band starts at the last recorded point so it fans out from known data
                datasets += [
                    {"label": "Range low", "data": pad + [hist["values"][-1]] + fc["lower"]},
                    {"label": "Likely range", "data": pad + [hist["values"][-1]] + fc["upper"]},
                ]
            return {
                "kind": "forecast", "title": "History + model forecast",
                "x_label": "Period", "y_label": unit,
                "labels": hist["labels"] + fc["labels"],
                "datasets": datasets,
                "forecast": fc,
            }
        # graph_recent (or anything else with history)
        n = max(1, min(len(history), ctx.last_n if ctx.last_n else (6 if billing else 30)))
        hist = _history_points(history, n, billing)
        return {
            "kind": "line", "title": f"Recent {'billing totals' if billing else 'daily supply'} (last {n})",
            "x_label": "Period", "y_label": unit,
            "labels": hist["labels"],
            "datasets": [{"label": "Recorded", "data": hist["values"]}],
        }
    except Exception:
        return None


# ---------------------------------------------------------------- markdown
_CMD_LINE = re.compile(r"^\s{2,}(.+?)\s{2,}->\s+(.+)$")
_NUM_UNIT = re.compile(r"(?<![\w*])([+-]?\d[\d,]*(?:\.\d+)?(?: lakh)? (?:QTY|m3(?:/day)?))")


MODEL_NAMES = {
    "cnn_lstm_attention": "CNN-LSTM + Attention",
    "lstm_attention": "LSTM + Attention",
    "cnn_lstm": "CNN-LSTM",
    "lstm": "LSTM",
    "naive_last_value": "Repeat last value",
    "moving_average_6": "Average of last 6 bills",
    "moving_average_7": "Average of last 7 days",
    "seasonal_naive_6": "Same bill last year",
    "seasonal_naive_7": "Same day last week",
}
# longest first so 'cnn_lstm_attention' is not partly replaced by 'lstm'
_MODEL_RE = re.compile(r"\b(" + "|".join(sorted(MODEL_NAMES, key=len, reverse=True)) + r")\b")


def _model_name(key: str) -> str:
    return MODEL_NAMES.get(key, key.replace("_", " "))


def _friendly_names(text: str) -> str:
    """Show readable model names in replies instead of internal ids like 'lstm_attention'."""
    return _MODEL_RE.sub(lambda m: MODEL_NAMES[m.group(1)], text)


def _to_markdown(text: str) -> str:
    """Turn the terminal-style replies from chatbot.py into light markdown for the web UI.

    Indented lines become bullets, 'cmd  -> meaning' help rows become '`cmd` — meaning',
    a heading line ending in ':' above a list is bolded, and quantities are emphasised.
    """
    lines = text.rstrip().split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        cmd = _CMD_LINE.match(line)
        if cmd:
            out.append(f"- `{cmd.group(1).strip()}` — {cmd.group(2).strip()}")
            continue
        if line.startswith("  ") and line.strip():
            out.append("- " + _NUM_UNIT.sub(r"**\1**", line.strip()))
            continue
        if line.rstrip().endswith(":") and nxt.startswith("  ") and not line.startswith("- "):
            out.append(f"**{line.strip()[:-1]}**")
            out.append("")
            continue
        out.append(_NUM_UNIT.sub(r"**\1**", line) if line.strip() else line)
    return "\n".join(out).replace(" -> ", " → ")


# ---------------------------------------------------------------- schemas
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    run_id: str | None = None
    session_id: str | None = None
    lang: str | None = Field(default=None, pattern="^(auto|en|hinglish)?$")


class ChatResponse(BaseModel):
    reply: str
    run_id: str
    session_id: str
    lang: str
    topic: str | None = None
    chart: dict | None = None
    forecast: dict | None = None
    history: dict | None = None


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health() -> dict:
    runs = _available_runs()
    return {"status": "ok", "runs": len(runs)}


@app.get("/api/runs")
def list_runs() -> dict:
    runs = [_run_summary(p) for p in _available_runs()]
    # newest first (folder names end with timestamp)
    runs.sort(key=lambda r: r["id"], reverse=True)
    latest = find_latest_run(OUTPUTS)
    return {"runs": runs, "default_run": latest.name if latest else None}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    rid, bundle = _load_run(run_id)
    summary = _run_summary(bundle["dir"])
    hist = _history_points(bundle["history"], min(8, len(bundle["history"])),
                           bundle["meta"].get("data_frequency") == "two_month_billing_period")
    return {"run": summary, "metrics": _metrics_rows(bundle), "recent": hist}


@app.get("/api/history")
def api_history(run_id: str | None = None, count: int = Query(8, ge=1, le=365)) -> dict:
    rid, bundle = _load_run(run_id)
    billing = bundle["meta"].get("data_frequency") == "two_month_billing_period"
    pts = _history_points(bundle["history"], count, billing)
    return {"run_id": rid, "unit": "QTY" if billing else "m3", **pts}


@app.get("/api/forecast")
def api_forecast(run_id: str | None = None, count: int = Query(1, ge=1, le=60)) -> dict:
    rid, bundle = _load_run(run_id)
    return {"run_id": rid, **_forecast_points(bundle, count)}


@app.get("/api/metrics")
def api_metrics(run_id: str | None = None) -> dict:
    rid, bundle = _load_run(run_id)
    return {"run_id": rid, "metrics": _metrics_rows(bundle)}


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    # Resolve run: explicit > session's run > latest
    run_id = req.run_id or (_SESSION_RUN.get(req.session_id) if req.session_id else None)
    rid, bundle = _load_run(run_id)
    sid = req.session_id or uuid.uuid4().hex[:12]
    ctx = _SESSIONS.get(sid)
    if ctx is None:
        lang_mode = {"auto": "auto", "en": "en", "hinglish": "hi"}.get(req.lang or "auto", "auto")
        ctx = bot.ChatContext(lang_mode=lang_mode)
        _SESSIONS[sid] = ctx
        _SESSION_RUN[sid] = rid
    elif req.run_id and req.run_id != _SESSION_RUN.get(sid):
        # switching runs mid-session: fresh context to avoid cross-run confusion
        ctx = bot.ChatContext(lang_mode=ctx.lang_mode)
        _SESSIONS[sid] = ctx
    _SESSION_RUN[sid] = rid
    if req.lang:
        # the UI language toggle applies to every message, not only the first one in a session
        ctx.lang_mode = {"auto": "auto", "en": "en", "hinglish": "hi"}[req.lang]

    # Snapshot graph state so sequential questions don't leak old charts
    ctx.last_graph_kind = None
    ctx.last_graph_describe = None
    ctx.topic_fresh = False

    try:
        reply, ctx = bot.answer(
            req.message, bundle["model"], bundle["scaler"], bundle["meta"],
            bundle["history"], bundle["metrics"], bundle["dir"], ctx, open_graph=False,
        )
    except Exception as exc:
        raise HTTPException(500, f"Chat failed ({type(exc).__name__}: {exc})")
    if reply == "EXIT":
        reply = "Goodbye! Type a new question anytime to keep chatting."

    chart = _build_chart(bundle, ctx)
    if chart is not None and ("Saved graph:" in reply or "Graph save ho gaya:" in reply):
        # The terminal message contains a local PNG path; the web UI renders
        # the same data interactively, so keep only the human-readable part.
        parts = reply.split("\n\n", 1)
        reading = parts[1] if len(parts) == 2 else parts[0]
        prefix = "Here's the chart — hover any point for exact values.\n\n"
        reply = prefix + reading.strip()
    reply = _friendly_names(_to_markdown(reply))
    # chatbot.py describes the terminal's PNG colours; the web chart uses a solid line, a dashed line and a band
    reply = reply.replace("Black line = recorded history. Red line = the model's estimate (not a reading).",
                          "Solid line = recorded history. Dashed line = the model's estimate (not a reading); "
                          "the shaded band is its likely range.")
    reply = reply.replace("Kaali line = recorded history. Laal line = model ka anuman (asli reading nahi).",
                          "Seedhi line = recorded history. Dashed line = model ka anuman (asli reading nahi); "
                          "shaded hissa uski andaazan range hai.")
    topic = ctx.last_topic if ctx.topic_fresh else None  # attach data only for what this message asked
    payload: dict = {
        "reply": reply, "run_id": rid, "session_id": sid,
        "lang": ctx.lang, "topic": topic, "chart": chart,
        "forecast": None, "history": None,
    }
    # Attach data tables for common topics so the UI can render cards
    try:
        billing = bundle["meta"].get("data_frequency") == "two_month_billing_period"
        if topic == "forecast":
            payload["forecast"] = _forecast_points(bundle, ctx.last_forecast_days if not billing else max(1, min(6, ctx.last_forecast_days)))
        elif topic == "recent":
            payload["history"] = _history_points(bundle["history"], max(1, min(len(bundle["history"]), ctx.last_n)), billing)
    except Exception:
        pass
    return ChatResponse(**payload)
