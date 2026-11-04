#!/usr/bin/env python3
"""Generate a synthetic dataset with the same schema as the original data.

The original store-level dataset is not redistributed. This script produces a
realistic stand-in so anyone can run the pipeline end to end. It is built from
the documented data-generating assumptions below (no rows are copied), so the
numbers you get from it will be close to, but not the same as, the figures in
the README.

    python scripts/generate_sample_data.py --stores 50 --out data/sample_sales.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# suburb: (stores in the network, population density /km², bachelor share, median income 2019)
SUBURBS = {
    "Newtown":       (7, 5771, 0.551, 124_000),
    "Circular Quay": (7, 7198, 0.462, 97_600),
    "Chatswood":     (7, 6144, 0.505, 98_900),
    "Bondi Beach":   (6, 7956, 0.492, 109_800),
    "Redfern":       (6, 5082, 0.538, 121_000),
    "Surry Hills":   (6, 6593, 0.521, 114_300),
    "Manly":         (4, 8089, 0.497, 109_400),
    "CBD Wynyard":   (3, 7648, 0.476, 107_200),
    "Parramatta":    (2, 8114, 0.433, 109_800),
    "Ultimo":        (2, 5278, 0.544, 120_000),
}
SEASON_OF_MONTH = {12: "Summer", 1: "Summer", 2: "Summer", 3: "Autumn", 4: "Autumn", 5: "Autumn",
                   6: "Winter", 7: "Winter", 8: "Winter", 9: "Spring", 10: "Spring", 11: "Spring"}
TEMP = {"Summer": (22.7, 1.7), "Autumn": (20.6, 2.5), "Winter": (13.3, 1.7), "Spring": (15.2, 2.5)}
SEASON_SHIFT = {"Summer": -20, "Autumn": 0, "Winter": -25, "Spring": -75}  # bottles vs autumn
PROMO_LEVELS = [1500, 3000, 5000, 8000]
PROMO_EFFECT = {0: 0, 1500: 6, 3000: 9, 5000: 16, 8000: 27}           # extra bottles


def generate(n_stores: int = 50, seed: int = 7, start="2019-01-06", end="2025-09-28") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="W-SUN")

    # Allocate stores to suburbs in proportion to the real network.
    weights = np.array([v[0] for v in SUBURBS.values()], float)
    alloc = np.maximum(1, np.round(weights / weights.sum() * n_stores)).astype(int)
    stores = []
    for (suburb, info), k in zip(SUBURBS.items(), alloc):
        for i in range(k):
            stores.append((suburb, info, i + 1))
    stores = stores[:n_stores]

    frames = []
    for sid, (suburb, (_, density, edu, income0), idx) in enumerate(stores, 1):
        base = rng.normal(300, 23)                                   # store's own level
        n = len(dates)
        month = dates.month.to_numpy()
        season = np.array([SEASON_OF_MONTH[m] for m in month])
        temp = np.array([rng.normal(*TEMP[s]) for s in season]).round(1)
        holiday = np.isin(month, [12, 1]).astype(int)
        promo = np.where(rng.random(n) < 0.2, rng.choice(PROMO_LEVELS, n), 0)
        year = dates.year.to_numpy()

        units = (base
                 + np.vectorize(SEASON_SHIFT.get)(season)
                 + (9.77 * temp - 0.214 * temp ** 2) - (9.77 * 18 - 0.214 * 18 ** 2)  # peaks ≈ 22.8°C
                 + 103 * holiday
                 + np.vectorize(PROMO_EFFECT.get)(promo)
                 + 3 * (year - 2019)
                 + rng.normal(0, 36, n))
        spikes = rng.random(n) < 0.004                                # rare data-entry spikes
        units = np.where(spikes, units * 1.9, units)
        units = np.clip(units.round(), 20, None).astype(int)
        price = np.clip(rng.normal(4.5, 0.086, n), 4.2, 4.8).round(2)

        frames.append(pd.DataFrame({
            "store_id": f"SDS-{sid:03d}",
            "store_name": f"Sample Store – {suburb} #{idx}",
            "suburb": suburb,
            "date": dates.strftime("%-d/%m/%Y"),
            "year": year,
            "week": dates.isocalendar().week.to_numpy(),
            "month": month,
            "season": season,
            "sales_units": units,
            "price_per_unit": price,
            "sales_revenue": (units * price).round(2),
            "avg_temp_c": temp,
            "promo_spend": promo,
            "holiday_flag": holiday,
            "median_income": (income0 + np.linspace(0, 9000, n)).round().astype(int),
            "education_bachelor_pct": edu,
            "population_density": density,
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stores", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/sample_sales.csv")
    a = p.parse_args()
    df = generate(a.stores, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} rows for {df['store_id'].nunique()} stores to {a.out}")


if __name__ == "__main__":
    main()
