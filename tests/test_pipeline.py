import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from generate_sample_data import generate  # noqa: E402
from sales_drivers import analysis as A  # noqa: E402
from sales_drivers.cleaning import clean, iqr_bounds, set_types  # noqa: E402
from sales_drivers.regression import design_matrix, ols_clustered  # noqa: E402


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return generate(n_stores=12, seed=1)


# ---------------------------------------------------------------- cleaning
def test_dates_parse_day_first_and_iso():
    df = pd.DataFrame({"date": ["6/01/2019", "2019-01-13"], "season": ["Summer", "Summer"],
                       "store_id": ["a", "a"], "suburb": ["x", "x"], "store_name": ["s", "s"],
                       "holiday_flag": [1, 1]})
    out = set_types(df)
    assert list(out["date"]) == [pd.Timestamp("2019-01-06"), pd.Timestamp("2019-01-13")]


def test_iqr_bounds_match_definition():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    lo, hi = iqr_bounds(s)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    assert (lo, hi) == (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))


def test_clean_removes_duplicates_missing_and_outliers(sample):
    dirty = pd.concat([sample, sample.head(3)], ignore_index=True)       # 3 duplicates
    dirty.loc[5, "avg_temp_c"] = np.nan                                   # 1 missing (not a duplicated row)
    dirty.loc[10, "sales_units"] = 50_000                                 # 1 obvious outlier
    df, report = clean(dirty)
    steps = {name: removed for name, _, removed in report.steps}
    assert steps["Drop missing values"] == 1
    assert steps["Drop duplicate rows"] == 3
    assert 50_000 not in df["sales_units"].values
    assert report.final_rows == len(df)


# ---------------------------------------------------------------- regression
def test_ols_matches_numpy_lstsq():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=500), "g": rng.integers(0, 25, 500)})
    df["y"] = 2.0 + 3.0 * df["x"] + rng.normal(size=500)
    X = design_matrix(df, numeric=["x"])
    res = ols_clustered(df["y"], X, df["g"])
    expected, *_ = np.linalg.lstsq(X.to_numpy(), df["y"].to_numpy(), rcond=None)
    np.testing.assert_allclose(res.params["coef"].to_numpy(), expected, rtol=1e-8)
    assert (res.params["se"] > 0).all()


def test_clustered_se_wider_than_naive_when_errors_are_clustered():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(40), 50)
    x = np.repeat(rng.normal(size=40), 50)            # regressor constant within cluster
    y = 1 + 0.5 * x + np.repeat(rng.normal(size=40), 50) + rng.normal(size=g.size)
    df = pd.DataFrame({"x": x, "y": y, "g": g})
    X = design_matrix(df, numeric=["x"])
    clustered = ols_clustered(df["y"], X, df["g"])["x"]["se"]
    naive = ols_clustered(df["y"], X, pd.Series(np.arange(len(df))))["x"]["se"]
    assert clustered > 2 * naive


# ---------------------------------------------------------------- analysis
def test_driver_model_recovers_known_effects(sample):
    df, _ = clean(sample)
    m = A.driver_model(df)
    assert m["holiday_flag"]["ci_low"] < 103 < m["holiday_flag"]["ci_high"] + 10
    assert 19 < A.peak_temperature(m) < 27
    roi = A.promo_roi_table(df, m)
    assert list(roi["promo_spend"]) == [1500, 3000, 5000, 8000]
    assert roi["extra_units"].iloc[-1] > roi["extra_units"].iloc[0]


def test_per_store_comparison_removes_store_count_bias(sample):
    df, _ = clean(sample)
    comp = A.suburb_comparison(df)
    spread_summed = comp["summed_weekly_units"].max() / comp["summed_weekly_units"].min()
    spread_per_store = comp["per_store_weekly_units"].max() / comp["per_store_weekly_units"].min()
    assert spread_per_store < spread_summed


def test_full_pipeline_writes_report_and_figures(tmp_path, sample):
    from run_analysis import run
    path = tmp_path / "data.csv"
    sample.to_csv(path, index=False)
    results = run(str(path), tmp_path / "out")
    assert (tmp_path / "out" / "results.md").exists()
    assert len(results["figures"]) == 5 and all(p.exists() for p in results["figures"])
