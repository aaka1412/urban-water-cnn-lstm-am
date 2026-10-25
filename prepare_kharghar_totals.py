"""Build one Kharghar QTY total per complete two-month billing period.

Usage:
  python prepare_kharghar_totals.py

The source workbooks contain a complete customer list for one period and small,
zero-QTY fragments for a neighboring period. This script keeps the largest
snapshot for each period and omits those incomplete fragments.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
SOURCE_DIR = ROOT / "data" / "KHARGHAR EXCELSHEETS"
OUTPUT_PATH = ROOT / "data" / "kharghar_period_totals.csv"
MIN_COMPLETE_ROWS = 500
MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JLY": 7, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def main() -> None:
    workbooks = sorted(SOURCE_DIR.glob("*.xls"))
    if not workbooks:
        raise FileNotFoundError(f"No .xls workbooks found in {SOURCE_DIR}")

    snapshots: list[dict] = []
    fragments: list[dict] = []
    for path in workbooks:
        frame = pd.read_excel(path, engine="xlrd")
        required = {"PERIOD", "CONSUMER NO", "QTY"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame["PERIOD"] = frame["PERIOD"].astype("string").str.strip()
        frame["QTY"] = pd.to_numeric(frame["QTY"], errors="coerce")
        if frame["QTY"].isna().any():
            raise ValueError(f"Non-numeric QTY values found in {path.name}")

        for period, group in frame.groupby("PERIOD", dropna=False):
            if pd.isna(period) or not str(period).strip():
                raise ValueError(f"Blank PERIOD value found in {path.name}")
            record = {
                "billing_period": str(period),
                "source_file": path.name,
                "customer_rows": len(group),
                "unique_customers": group["CONSUMER NO"].nunique(dropna=True),
                "total_qty": float(group["QTY"].sum()),
            }
            (snapshots if len(group) >= MIN_COMPLETE_ROWS else fragments).append(record)

    if any(record["total_qty"] != 0 for record in fragments):
        raise ValueError("A partial-period fragment has nonzero QTY; review it before summing.")

    totals = pd.DataFrame(snapshots)
    if totals.empty:
        raise ValueError("No complete billing-period groups were found.")
    repeated = totals[totals.duplicated("billing_period", keep=False)]
    if not repeated.empty:
        raise ValueError(
            "Multiple complete snapshots exist for a billing period; choose one before summing:\n"
            + repeated.to_string(index=False)
        )
    if (totals["customer_rows"] != totals["unique_customers"]).any():
        raise ValueError("Duplicate customer rows found in a complete billing-period group.")

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for label in totals["billing_period"]:
        match = re.fullmatch(r"([A-Z]{3})-([A-Z]{3})(\d{4})", label)
        if not match or match.group(1) not in MONTHS or match.group(2) not in MONTHS:
            raise ValueError(f"Cannot parse billing-period label: {label}")
        start_month, end_month, end_year = MONTHS[match.group(1)], MONTHS[match.group(2)], int(match.group(3))
        start_year = end_year - 1 if start_month == 12 else end_year
        starts.append(pd.Timestamp(start_year, start_month, 1))
        ends.append(pd.Timestamp(end_year, end_month, 1) + pd.offsets.MonthEnd(0))

    totals["period_start"] = starts
    totals["period_end"] = ends
    totals = totals.sort_values("period_start").reset_index(drop=True)
    if totals["period_start"].duplicated().any():
        raise ValueError("More than one billing period maps to the same start date.")

    result = totals[
        ["period_start", "period_end", "billing_period", "customer_rows", "unique_customers", "total_qty", "source_file"]
    ].copy()
    result["period_start"] = result["period_start"].dt.strftime("%Y-%m-%d")
    result["period_end"] = result["period_end"].dt.strftime("%Y-%m-%d")
    result.to_csv(OUTPUT_PATH, index=False, float_format="%.2f")
    print(f"Created {OUTPUT_PATH.relative_to(ROOT)} with {len(result)} complete billing periods.")
    print(f"Coverage: {result.iloc[0]['billing_period']} through {result.iloc[-1]['billing_period']}.")
    print(f"Omitted {len(fragments)} partial-period groups; all had zero QTY.")
    print("QTY unit is carried through as provided by the source; it has not been confirmed.")


if __name__ == "__main__":
    main()
