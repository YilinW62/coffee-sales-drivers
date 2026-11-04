"""Descriptive analysis and driver models for the three business questions.

Q1  How much do promotions lift weekly sales, and do they pay back?
Q2  How do season and temperature shape demand?
Q3  Do some suburbs genuinely outperform others?

All "per store" figures are averages over store-weeks, so suburbs with more
stores are not mechanically favoured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import SEASON_ORDER
from .regression import OLSResult, design_matrix, ols_clustered

CONTROLS_NUMERIC = ["avg_temp_c", "temp_sq", "holiday_flag"]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_sq"] = df["avg_temp_c"] ** 2
    df["promo"] = (df["promo_spend"] > 0).astype(int)
    return df


# --------------------------------------------------------------------------- Q1
def promo_lift(df: pd.DataFrame) -> dict:
    """Raw comparison of promotion vs non-promotion store-weeks."""
    promo = df["promo_spend"] > 0
    with_p, without_p = df.loc[promo, "sales_units"].mean(), df.loc[~promo, "sales_units"].mean()
    return {
        "promo_share_of_weeks": promo.mean(),
        "units_with_promo": with_p,
        "units_without_promo": without_p,
        "raw_lift_pct": (with_p / without_p - 1) * 100,
    }


def spend_by_season(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("season", observed=True)
    return pd.DataFrame({
        "total_promo_spend": g["promo_spend"].sum(),
        "promo_week_share": g["promo_spend"].apply(lambda s: (s > 0).mean()),
    }).reindex(SEASON_ORDER)


def driver_model(df: pd.DataFrame) -> OLSResult:
    """Weekly units ~ promo budget level + season + temperature (quadratic)
    + holiday + store fixed effects + year fixed effects."""
    d = _prep(df)
    d["promo_level"] = d["promo_spend"].astype(int)
    X = design_matrix(d, numeric=CONTROLS_NUMERIC,
                      categorical=["promo_level", "season", "store_id", "year"])
    return ols_clustered(d["sales_units"], X, d["store_id"])


def promo_roi_table(df: pd.DataFrame, model: OLSResult) -> pd.DataFrame:
    """Incremental units and same-week revenue returned per promotional dollar."""
    price = df["price_per_unit"].mean()
    rows = []
    for name in model.params.index:
        if name.startswith("promo_level_"):
            spend = int(name.removeprefix("promo_level_"))
            r = model[name]
            rows.append({
                "promo_spend": spend,
                "extra_units": r["coef"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                "extra_revenue": r["coef"] * price,
                "revenue_per_promo_dollar": r["coef"] * price / spend,
            })
    return pd.DataFrame(rows).sort_values("promo_spend").reset_index(drop=True)


def promo_effect_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Promotion lift (units) estimated separately for each season."""
    d = _prep(df)
    extra = {f"promo_x_{s}": d["promo"] * (d["season"] == s) for s in SEASON_ORDER}
    X = design_matrix(d, numeric=CONTROLS_NUMERIC,
                      categorical=["season", "store_id", "year"], extra=extra)
    m = ols_clustered(d["sales_units"], X, d["store_id"])
    out = m.params.loc[list(extra)].copy()
    out.index = SEASON_ORDER
    return out[["coef", "ci_low", "ci_high"]]


# --------------------------------------------------------------------------- Q2
def season_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("season", observed=True)
    return pd.DataFrame({
        "avg_temp_c": g["avg_temp_c"].mean(),
        "mean_weekly_revenue": g["sales_revenue"].mean(),
        "median_weekly_revenue": g["sales_revenue"].median(),
        "holiday_week_share": g["holiday_flag"].mean(),
    }).reindex(SEASON_ORDER)


def temperature_curve(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Mean weekly revenue per store within equal-count temperature bins."""
    d = df.copy()
    d["temp_bin"] = pd.qcut(d["avg_temp_c"], q=bins, duplicates="drop")
    g = d.groupby("temp_bin", observed=True)
    return pd.DataFrame({
        "temp_mid": g["avg_temp_c"].mean(),
        "mean_weekly_revenue": g["sales_revenue"].mean(),
        "n": g.size(),
    }).reset_index(drop=True)


def peak_temperature(model: OLSResult) -> float:
    """Vertex of the fitted quadratic: b1*T + b2*T^2 peaks at -b1 / (2*b2)."""
    b1, b2 = model["avg_temp_c"]["coef"], model["temp_sq"]["coef"]
    return -b1 / (2 * b2)


def season_temperature_grid(df: pd.DataFrame) -> pd.DataFrame:
    bands = [-np.inf, 12, 16, 20, 24, 28, np.inf]
    labels = ["<12", "12–16", "16–20", "20–24", "24–28", "≥28"]
    d = df.copy()
    d["temp_band"] = pd.cut(d["avg_temp_c"], bins=bands, labels=labels, right=False)
    return (d.pivot_table(index="season", columns="temp_band", values="sales_revenue",
                          aggfunc="mean", observed=False)
              .reindex(index=SEASON_ORDER, columns=labels))


# --------------------------------------------------------------------------- Q3
def suburb_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Contrast the naive suburb total with the fair per-store average."""
    stores = df.groupby("suburb", observed=True)["store_id"].nunique()
    summed = (df.groupby(["suburb", "year", "week"], observed=True)["sales_units"].sum()
                .groupby("suburb", observed=True).mean())
    per_store = df.groupby("suburb", observed=True)["sales_units"].mean()
    out = pd.DataFrame({"stores": stores, "summed_weekly_units": summed,
                        "per_store_weekly_units": per_store})
    out["population_density"] = df.groupby("suburb", observed=True)["population_density"].mean()
    return out.sort_values("per_store_weekly_units", ascending=False)


def suburb_model(df: pd.DataFrame) -> pd.DataFrame:
    """Suburb effects on per-store weekly units, holding promo, season,
    temperature, holiday and year constant (no store fixed effects, since
    stores are nested in suburbs). Reference suburb = first alphabetically."""
    d = _prep(df)
    X = design_matrix(d, numeric=["promo", *CONTROLS_NUMERIC],
                      categorical=["suburb", "season", "year"])
    m = ols_clustered(d["sales_units"], X, d["store_id"])
    out = m.params.filter(like="suburb_", axis=0)[["coef", "ci_low", "ci_high", "t"]].copy()
    out.index = out.index.str.removeprefix("suburb_")
    return out.sort_values("coef")


def store_spread(df: pd.DataFrame) -> pd.Series:
    return df.groupby("store_id", observed=True)["sales_units"].mean().describe()
