"""Load and clean the weekly store-level sales dataset.

Cleaning steps (each one is logged so the retention table can be reported):
    1. parse types (dates, categoricals, flags)
    2. drop rows with missing values
    3. drop exact duplicate rows
    4. remove outliers with the 1.5 x IQR rule, column by column
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EXPECTED_COLUMNS = [
    "store_id", "store_name", "suburb", "date", "year", "week", "month", "season",
    "sales_units", "price_per_unit", "sales_revenue", "avg_temp_c", "promo_spend",
    "holiday_flag", "median_income", "education_bachelor_pct", "population_density",
]

# Columns screened for outliers. promo_spend is excluded on purpose: it takes a
# handful of planned budget levels ($0 / 1.5k / 3k / 5k / 8k), so "extreme"
# values are business decisions, not measurement errors.
OUTLIER_COLUMNS = [
    "sales_units", "sales_revenue", "avg_temp_c", "median_income", "population_density",
]

SEASON_ORDER = ["Summer", "Autumn", "Winter", "Spring"]  # Southern Hemisphere, Dec-Feb = Summer


@dataclass
class CleaningReport:
    original_rows: int
    steps: list[tuple[str, int, int]] = field(default_factory=list)  # (step, rows_after, removed)
    outliers: dict[str, tuple[int, float, float]] = field(default_factory=dict)

    @property
    def final_rows(self) -> int:
        return self.steps[-1][1] if self.steps else self.original_rows

    def to_markdown(self) -> str:
        lines = ["| Step | Rows | Removed | Retained |", "|---|---:|---:|---:|",
                 f"| Original | {self.original_rows:,} | – | 100.00% |"]
        for name, rows, removed in self.steps:
            lines.append(f"| {name} | {rows:,} | {removed:,} | {rows / self.original_rows:.2%} |")
        return "\n".join(lines)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df[EXPECTED_COLUMNS].copy()


def set_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Source files use Australian day-first dates (6/01/2019); ISO dates also parse.
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
    df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)
    for col in ["store_id", "suburb"]:
        df[col] = df[col].astype("category")
    df["store_name"] = df["store_name"].astype("string")
    df["holiday_flag"] = df["holiday_flag"].astype(int)
    return df


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(original_rows=len(df))
    df = set_types(df)

    before = len(df)
    df = df.dropna()
    report.steps.append(("Drop missing values", len(df), before - len(df)))

    before = len(df)
    df = df.drop_duplicates()
    report.steps.append(("Drop duplicate rows", len(df), before - len(df)))

    before = len(df)
    for col in OUTLIER_COLUMNS:
        lo, hi = iqr_bounds(df[col])
        mask = df[col].between(lo, hi)
        report.outliers[col] = (int((~mask).sum()), float(lo), float(hi))
        df = df[mask]
    report.steps.append(("Remove IQR outliers", len(df), before - len(df)))

    return df.reset_index(drop=True), report
