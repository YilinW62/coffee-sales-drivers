#!/usr/bin/env python3
"""Run the full pipeline: load -> clean -> analyse -> model -> figures -> report.

    python run_analysis.py --data data/sample_sales.csv --out reports
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from sales_drivers import analysis as A  # noqa: E402
from sales_drivers import plots  # noqa: E402
from sales_drivers.cleaning import clean, load_raw  # noqa: E402


def run(data_path: str, out_dir: Path) -> dict:
    raw = load_raw(data_path)
    df, cleaning = clean(raw)

    model = A.driver_model(df)
    results = {
        "cleaning": cleaning,
        "promo_lift": A.promo_lift(df),
        "spend_by_season": A.spend_by_season(df),
        "driver_model": model,
        "promo_roi": A.promo_roi_table(df, model),
        "promo_by_season": A.promo_effect_by_season(df),
        "season_summary": A.season_summary(df),
        "temperature_curve": A.temperature_curve(df),
        "peak_temperature": A.peak_temperature(model),
        "season_temperature_grid": A.season_temperature_grid(df),
        "suburb_comparison": A.suburb_comparison(df),
        "suburb_model": A.suburb_model(df),
        "store_spread": A.store_spread(df),
    }
    figures = plots.make_all(df, results, out_dir / "figures")
    (out_dir / "results.md").write_text(render_report(df, results))
    results["figures"] = figures
    return results


def _md(frame: pd.DataFrame, floatfmt: str = ",.2f") -> str:
    return frame.to_markdown(floatfmt=floatfmt) if hasattr(frame, "to_markdown") else frame.to_string()


def render_report(df: pd.DataFrame, r: dict) -> str:
    m, lift = r["driver_model"], r["promo_lift"]
    hol = m["holiday_flag"]
    base = df.loc[df["promo_spend"] == 0, "sales_units"].mean()
    out = [
        "# Results",
        f"Rows analysed: {len(df):,} store-weeks · {df['store_id'].nunique()} stores · "
        f"{df['suburb'].nunique()} suburbs · {df['date'].min():%b %Y} – {df['date'].max():%b %Y}",
        "", "## Data cleaning", r["cleaning"].to_markdown(), "",
        "Outliers removed per column: " + ", ".join(
            f"{c} {n}" for c, (n, _, _) in r["cleaning"].outliers.items() if n),
        "", "## Q1 Promotions",
        f"- Raw comparison: {lift['units_with_promo']:.1f} vs {lift['units_without_promo']:.1f} "
        f"bottles/week (+{lift['raw_lift_pct']:.1f}%), promotions run in {lift['promo_share_of_weeks']:.0%} of store-weeks",
        "", "Regression estimate by budget level:", "",
        _md(r["promo_roi"].set_index("promo_spend"), ",.3f"), "",
        "Promotion lift by season (extra bottles/week):", "", _md(r["promo_by_season"]), "",
        "Spend by season:", "", _md(r["spend_by_season"], ",.3f"),
        "", "## Q2 Season and temperature",
        f"- Holiday weeks: +{hol['coef']:.1f} bottles/week (95% CI {hol['ci_low']:.1f}–{hol['ci_high']:.1f}), "
        f"{hol['coef'] / base:.0%} of a normal week",
        f"- Fitted temperature peak: {r['peak_temperature']:.1f}°C", "",
        _md(r["season_summary"]), "", _md(r["temperature_curve"]), "",
        "## Q3 Suburbs", "", _md(r["suburb_comparison"], ",.1f"), "",
        "Suburb effects vs reference suburb (bottles/week, controls held constant):", "",
        _md(r["suburb_model"]), "",
        "Store-level average weekly bottles:", "", _md(r["store_spread"].to_frame("units"), ",.1f"), "",
        f"Driver model: R² = {m.r2:.3f}, n = {m.n:,}, clusters = {m.n_clusters}",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/sample_sales.csv")
    p.add_argument("--out", default="reports")
    args = p.parse_args()
    results = run(args.data, Path(args.out))
    print((Path(args.out) / "results.md").read_text())
    print("Figures:", *map(str, results["figures"]), sep="\n  ")


if __name__ == "__main__":
    main()
