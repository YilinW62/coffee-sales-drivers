"""Static figures for the report (matplotlib only).

Style: one accent hue for single-series charts, recessive grid and axes,
direct labels on the values that matter, one y-axis per chart.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .cleaning import SEASON_ORDER  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SEQ_BLUES = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 14, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False,
})


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=10, va="bottom")


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def promo_roi(roi: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(roi))
    ax.bar(x, roi["extra_units"], width=0.55, color=BLUE, zorder=2)
    ax.errorbar(x, roi["extra_units"],
                yerr=[roi["extra_units"] - roi["ci_low"], roi["ci_high"] - roi["extra_units"]],
                fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=4, zorder=3)
    for xi, (_, r) in zip(x, roi.iterrows()):
        ax.text(xi, r["ci_high"] + 0.8, f"+{r['extra_units']:.1f} bottles\n"
                f"${r['extra_revenue']:.0f} revenue\n{r['revenue_per_promo_dollar']*100:.1f}¢ per $1",
                ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set_xticks(x, [f"${s:,.0f}" for s in roi["promo_spend"]])
    ax.set_xlabel("Weekly promotional spend per store")
    ax.set_ylabel("Extra bottles sold vs. no promotion")
    ax.set_ylim(0, roi["ci_high"].max() * 1.65)
    ax.grid(axis="x", visible=False)
    ax.set_title("Promotions lift sales, but return about 2¢ per dollar", pad=26)
    _subtitle(ax, "Regression estimate with 95% CI, controlling for store, season, temperature, holidays and year")
    return _save(fig, out_dir, "promo_roi.png")


def season_revenue(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    data = [df.loc[df["season"] == s, "sales_revenue"] for s in SEASON_ORDER]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.6},
                    whiskerprops={"color": AXIS}, capprops={"color": AXIS},
                    boxprops={"edgecolor": SURFACE, "linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.85)
    labels = [f"{s}\navg {summary.loc[s, 'avg_temp_c']:.1f}°C\n{summary.loc[s, 'holiday_week_share']:.0%} holiday weeks"
              for s in SEASON_ORDER]
    ax.set_xticks(range(1, 5), labels)
    for i, s in enumerate(SEASON_ORDER, 1):
        ax.text(i + 0.3, summary.loc[s, "median_weekly_revenue"],
                f"${summary.loc[s, 'median_weekly_revenue']:,.0f}", va="center", fontsize=9, color=INK_2)
    ax.set_ylabel("Weekly revenue per store ($)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Summer sells best; spring is the weakest season", pad=26)
    _subtitle(ax, "Distribution of store-week revenue (box = middle 50%, line = median)")
    return _save(fig, out_dir, "season_revenue.png")


def temperature(curve: pd.DataFrame, peak: float, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(curve["temp_mid"], curve["mean_weekly_revenue"], color=BLUE, linewidth=2,
            marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.axvline(peak, color=ORANGE, linestyle="--", linewidth=1.5, zorder=2)
    ax.text(peak + 0.2, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06,
            f"Model peak ≈ {peak:.1f}°C", color=INK_2, fontsize=9.5)
    ax.set_xlabel("Average weekly temperature (°C, decile midpoints)")
    ax.set_ylabel("Mean weekly revenue per store ($)")
    ax.set_title("Demand climbs with temperature and peaks around 22–23°C", pad=26)
    _subtitle(ax, "Mean revenue in ten equal-sized temperature bins; peak from the quadratic regression term")
    return _save(fig, out_dir, "temperature_curve.png")


def season_temperature_heatmap(grid: pd.DataFrame, out_dir: Path) -> Path:
    grid = grid.loc[:, grid.notna().any()]  # drop bands with no weeks at all
    fig, ax = plt.subplots(figsize=(8, 3.8))
    v = grid.to_numpy(float)
    im = ax.imshow(np.ma.masked_invalid(v), cmap=SEQ_BLUES, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), grid.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    vmax = np.nanmax(v)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, f"${v[i, j]:,.0f}", ha="center", va="center", fontsize=9.5,
                        color="#ffffff" if v[i, j] > 0.75 * vmax else INK)
            else:
                ax.text(j, i, "–", ha="center", va="center", color=MUTED)
    ax.set_xlabel("Temperature band (°C)")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED)
    ax.set_title("Mild-to-warm summer weeks are the sales window", pad=26)
    _subtitle(ax, "Mean weekly revenue per store by season × temperature band ('–' = no such weeks)")
    return _save(fig, out_dir, "season_temperature_heatmap.png")


def suburb_correction(comp: pd.DataFrame, out_dir: Path) -> Path:
    comp = comp.sort_values("summed_weekly_units")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    y = np.arange(len(comp))
    labels = [f"{s} ({n})" for s, n in zip(comp.index, comp["stores"])]
    for ax, col, color, title in [
        (axes[0], "summed_weekly_units", MUTED, "Naive: total of all stores"),
        (axes[1], "per_store_weekly_units", BLUE, "Fair: average per store"),
    ]:
        ax.barh(y, comp[col], height=0.6, color=color, zorder=2)
        for yi, val in zip(y, comp[col]):
            ax.text(val, yi, f"  {val:,.0f}", va="center", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, comp[col].max() * 1.22)
        ax.set_xlabel("Weekly bottles sold")
    axes[0].set_yticks(y, labels)
    fig.suptitle("Suburb 'winners' were mostly suburbs with more stores", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK, y=1.04)
    fig.text(0.01, 0.97, "Number of stores in brackets. Per store, every suburb sells within ~10% of the others.",
             color=INK_2, fontsize=10, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "suburb_correction.png")


def make_all(df: pd.DataFrame, results: dict, out_dir: Path) -> list[Path]:
    return [
        promo_roi(results["promo_roi"], out_dir),
        season_revenue(df, results["season_summary"], out_dir),
        temperature(results["temperature_curve"], results["peak_temperature"], out_dir),
        season_temperature_heatmap(results["season_temperature_grid"], out_dir),
        suburb_correction(results["suburb_comparison"], out_dir),
    ]
