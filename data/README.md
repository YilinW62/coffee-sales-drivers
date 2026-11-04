# Data

## What's here

`sample_sales.csv` is a **synthetic** dataset (20 stores, 7,040 store-weeks) produced by
[`scripts/generate_sample_data.py`](../scripts/generate_sample_data.py). It has the same schema
and similar statistical behaviour as the original data, so the whole pipeline runs out of the box.
Results on it are close to, but not identical to, the figures in the main README.

The original dataset (50 stores, 17,600 store-weeks, Jan 2019 – Sep 2025) is not redistributed.
To reproduce the README exactly, put a file with the same columns anywhere and run
`python run_analysis.py --data path/to/file.csv`.

## Schema (one row = one store, one week)

| Column | Type | Description |
|---|---|---|
| `store_id`, `store_name` | text | Store identifier and display name |
| `suburb` | text | One of 10 Sydney suburbs; stores per suburb range from 2 to 7 |
| `date` | date (`d/mm/yyyy`) | Week ending (Sunday) |
| `year`, `week`, `month` | int | Calendar fields |
| `season` | text | Southern-hemisphere season (Dec–Feb = Summer) |
| `sales_units` | int | Bottles sold that week |
| `price_per_unit` | float | Average selling price (AUD) |
| `sales_revenue` | float | `sales_units × price_per_unit` |
| `avg_temp_c` | float | Average temperature that week (°C) |
| `promo_spend` | int | Promotional budget that week: 0, 1,500, 3,000, 5,000 or 8,000 AUD |
| `holiday_flag` | 0/1 | School / public holiday week (December and January) |
| `median_income` | int | Suburb median household income (AUD) |
| `education_bachelor_pct` | float | Share of suburb residents with a bachelor's degree |
| `population_density` | int | Suburb residents per km² |
