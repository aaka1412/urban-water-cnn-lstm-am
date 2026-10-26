"""Upload validation for billing-period datasets.

Rules mirror run_kharghar_experiment.load_period_data (required columns, parseable dates/numbers,
one row per period, no negatives, continuous two-month steps) plus checks the model needs
(at least `sequence_length` periods) and warnings for data far outside the training range.
Nothing is renamed, reordered, filled or invented to make a file pass.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PREVIEW_ROWS = 20


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}  # status: passed | warning | failed


def read_table(raw: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV or XLSX bytes. No pickles, no macros, no legacy .xls."""
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        text = raw.decode("utf-8-sig", errors="replace")
        return pd.read_csv(io.StringIO(text), nrows=config.MAX_UPLOAD_ROWS + 1)
    if ext == ".xlsx":
        if not raw.startswith(b"PK"):
            raise ValueError("The file is named .xlsx but is not a valid Excel workbook.")
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl", nrows=config.MAX_UPLOAD_ROWS + 1)
    raise ValueError("Only .csv and .xlsx files are accepted.")


def validate_frame(frame: pd.DataFrame, sequence_length: int, train_min: float | None = None,
                   train_max: float | None = None, project_min_periods: int | None = None) -> dict:
    """Return a validation report. report['status'] is passed | warning | failed."""
    project_min_periods = project_min_periods or sequence_length * 4
    checks: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []
    n_rows, n_cols = frame.shape
    columns = [{"name": str(c), "dtype": str(frame[c].dtype), "missing": int(frame[c].isna().sum())}
               for c in frame.columns]
    report = {"n_rows": int(n_rows), "n_cols": int(n_cols), "columns": columns, "checks": checks,
              "errors": errors, "warnings": warnings, "missing_values": int(frame.isna().sum().sum()),
              "duplicate_records": int(frame.duplicated().sum()), "invalid_dates": 0, "invalid_values": 0,
              "preview": []}

    if n_rows > config.MAX_UPLOAD_ROWS:
        errors.append(f"File has more than {config.MAX_UPLOAD_ROWS} rows.")
        checks.append(_check("Row limit", "failed", errors[-1]))
    else:
        checks.append(_check("Row limit", "passed", f"{n_rows} rows (limit {config.MAX_UPLOAD_ROWS})."))

    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in frame.columns]
    if missing_cols:
        msg = (f"Missing required column(s): {', '.join(missing_cols)}. Required: "
               f"{', '.join(config.REQUIRED_COLUMNS)}. Columns are not renamed automatically.")
        errors.append(msg)
        checks.append(_check("Required columns", "failed", msg))
        report["status"] = "failed"
        report["compatible"] = False
        return report
    checks.append(_check("Required columns", "passed", "All required columns are present."))

    start = pd.to_datetime(frame["period_start"], errors="coerce")
    end = pd.to_datetime(frame["period_end"], errors="coerce")
    qty = pd.to_numeric(frame["total_qty"], errors="coerce")
    bad_dates = int(start.isna().sum() + end.isna().sum())
    bad_qty = int(qty.isna().sum())
    report["invalid_dates"], report["invalid_values"] = bad_dates, bad_qty
    if bad_dates:
        errors.append(f"{bad_dates} period_start/period_end value(s) are missing or not valid dates.")
        checks.append(_check("Dates parse", "failed", errors[-1]))
    else:
        checks.append(_check("Dates parse", "passed", "All period dates are valid."))
    if bad_qty:
        errors.append(f"{bad_qty} total_qty value(s) are missing or not numeric.")
        checks.append(_check("Consumption values numeric", "failed", errors[-1]))
    else:
        checks.append(_check("Consumption values numeric", "passed", "All total_qty values are numeric."))

    valid = qty.dropna()
    if (valid < 0).any():
        errors.append(f"{int((valid < 0).sum())} negative total_qty value(s); negative consumption is not supported.")
        checks.append(_check("No negative values", "failed", errors[-1]))
    elif (valid == 0).any():
        warnings.append(f"{int((valid == 0).sum())} period(s) have total_qty = 0; please verify.")
        checks.append(_check("No negative values", "warning", warnings[-1]))
    else:
        checks.append(_check("No negative values", "passed", "No negative or zero values."))

    dup_period = int(frame["billing_period"].duplicated().sum() + start.duplicated().sum())
    if dup_period:
        errors.append("Each billing period must appear exactly once (duplicate billing_period or period_start found).")
        checks.append(_check("One row per period", "failed", errors[-1]))
    else:
        checks.append(_check("One row per period", "passed", "No duplicate periods."))
    if report["duplicate_records"]:
        warnings.append(f"{report['duplicate_records']} fully duplicated row(s).")

    if not errors:
        order = start.argsort(kind="stable")
        if not (order.to_numpy() == np.arange(len(order))).all():
            warnings.append("Rows were not in date order; they will be read in chronological order.")
        s_sorted = start.iloc[order].reset_index(drop=True)
        e_sorted = end.iloc[order].reset_index(drop=True)
        gaps = [(a, b) for a, b in zip(s_sorted, s_sorted.iloc[1:]) if b != a + pd.DateOffset(months=2)]
        if gaps:
            a, b = gaps[0]
            errors.append(f"Billing periods are not continuous: {a.date()} is followed by {b.date()} "
                          "(two-month steps are required; missing periods are not filled in).")
            checks.append(_check("Continuous two-month periods", "failed", errors[-1]))
        else:
            checks.append(_check("Continuous two-month periods", "passed", "Periods follow in two-month steps."))
        odd = int(sum(e != s + pd.DateOffset(months=2) - pd.Timedelta(days=1) for s, e in zip(s_sorted, e_sorted)))
        if odd:
            warnings.append(f"{odd} row(s) have a period_end that is not two months after period_start.")
            checks.append(_check("Period length", "warning", warnings[-1]))
        else:
            checks.append(_check("Period length", "passed", "Every period spans two calendar months."))

    if len(valid) < sequence_length:
        errors.append(f"The model reads {sequence_length} consecutive periods; the file has only {len(valid)} usable row(s).")
        checks.append(_check("Enough history for the model", "failed", errors[-1]))
    elif len(valid) < project_min_periods:
        warnings.append(f"{len(valid)} periods is enough to predict, but fewer than the {project_min_periods} "
                        "the project's own training script needs; it cannot be used to retrain or backtest reliably.")
        checks.append(_check("Enough history for the model", "warning", warnings[-1]))
    else:
        checks.append(_check("Enough history for the model", "passed",
                             f"{len(valid)} periods (model window {sequence_length})."))

    if train_min is not None and train_max is not None and len(valid):
        outside = int(((valid < train_min) | (valid > train_max)).sum())
        if outside:
            warnings.append(f"{outside} value(s) lie outside the range the model was trained on "
                            f"({train_min:,.0f} to {train_max:,.0f} Units). Predictions may be unreliable.")
            checks.append(_check("Similar to training data", "warning", warnings[-1]))
        else:
            checks.append(_check("Similar to training data", "passed", "All values fall inside the training range."))

    preview = frame.head(PREVIEW_ROWS).copy()
    report["preview"] = preview.astype(object).where(preview.notna(), None).astype(str).replace("None", "").to_dict("records")
    report["status"] = "failed" if errors else ("warning" if warnings else "passed")
    report["compatible"] = not errors
    return report


def to_billing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validated upload -> chronological frame with parsed types (no values changed)."""
    out = frame[config.REQUIRED_COLUMNS].copy()
    out["period_start"] = pd.to_datetime(out["period_start"])
    out["period_end"] = pd.to_datetime(out["period_end"])
    out["total_qty"] = pd.to_numeric(out["total_qty"])
    out["billing_period"] = out["billing_period"].astype(str)
    return out.sort_values("period_start").reset_index(drop=True)
