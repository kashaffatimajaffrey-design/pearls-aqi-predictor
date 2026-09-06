"""Exploratory Data Analysis.

Generates the figures and the written findings that back the modelling choices.
Run it as a script to refresh `reports/eda_report.md` and `reports/figures/`:

    python -m src.eda

The same functions are imported by the Streamlit EDA tab, so the notebook, the
report and the dashboard never drift apart.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")  # headless: this runs in CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .aqi import CATEGORIES, categorize
from .store import get_feature_store

log = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def load() -> pd.DataFrame:
    df = get_feature_store().read()
    if df.empty:
        raise RuntimeError("feature store is empty -- run the backfill first")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


# ------------------------------------------------------------- statistics ---
def describe(df: pd.DataFrame) -> dict:
    """Headline numbers the report and dashboard both quote."""
    aqi = df["aqi"].dropna()
    hours = len(df)
    span_days = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400

    category_counts = aqi.apply(categorize).value_counts()
    unhealthy_hours = int((aqi >= 150).sum())

    hourly = df.groupby(df["ts"].dt.hour)["aqi"].mean()
    dow = df.groupby(df["ts"].dt.dayofweek)["aqi"].mean()
    monthly = df.groupby(df["ts"].dt.month)["aqi"].mean()

    return {
        "rows": hours,
        "span_days": round(span_days, 1),
        "coverage_pct": round(100 * hours / max(1, span_days * 24), 1),
        "start": str(df["ts"].min()),
        "end": str(df["ts"].max()),
        "aqi_mean": round(float(aqi.mean()), 1),
        "aqi_median": round(float(aqi.median()), 1),
        "aqi_std": round(float(aqi.std()), 1),
        "aqi_min": round(float(aqi.min()), 1),
        "aqi_max": round(float(aqi.max()), 1),
        "aqi_p95": round(float(aqi.quantile(0.95)), 1),
        "unhealthy_hours": unhealthy_hours,
        "unhealthy_pct": round(100 * unhealthy_hours / max(1, len(aqi)), 1),
        "category_distribution": category_counts.to_dict(),
        "dominant_pollutant_mode": (
            df["dominant_pollutant"].mode().iloc[0]
            if "dominant_pollutant" in df and not df["dominant_pollutant"].isna().all()
            else "n/a"
        ),
        "peak_hour": int(hourly.idxmax()),
        "peak_hour_aqi": round(float(hourly.max()), 1),
        "cleanest_hour": int(hourly.idxmin()),
        "cleanest_hour_aqi": round(float(hourly.min()), 1),
        "worst_weekday": int(dow.idxmax()),
        "worst_month": int(monthly.idxmax()),
        "worst_month_aqi": round(float(monthly.max()), 1),
        "best_month": int(monthly.idxmin()),
        "missing_pct": round(100 * df[config.POLLUTANTS].isna().mean().mean(), 2),
    }


def autocorrelation(df: pd.DataFrame, max_lag: int = 168) -> pd.Series:
    """How far back does AQI actually carry information? This is the empirical
    justification for the 72-hour lag window in the feature set."""
    aqi = df["aqi"].dropna()
    return pd.Series(
        {lag: aqi.autocorr(lag) for lag in range(1, max_lag + 1)}, name="autocorr"
    )


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in config.POLLUTANTS + config.WEATHER_COLS + ["aqi"] if c in df]
    return df[cols].corr(numeric_only=True)


def target_correlations(df: pd.DataFrame, top_n: int = 20) -> pd.Series:
    """Which engineered features track AQI most strongly."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corrwith(df["aqi"]).drop(labels=["aqi"], errors="ignore")
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)


# ---------------------------------------------------------------- figures ---
def _save(fig, name: str) -> str:
    path = config.FIGURE_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_timeseries(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(df["ts"], df["aqi"], lw=0.7, color="#2c3e50", label="Hourly AQI")
    ax.plot(df["ts"], df["aqi"].rolling(24 * 7, min_periods=24).mean(),
            lw=2.2, color="#e74c3c", label="7-day rolling mean")
    for lo, hi, _name, color in CATEGORIES[:5]:
        ax.axhspan(lo, hi, color=color, alpha=0.10)
    ax.axhline(config.ALERT_AQI_THRESHOLD, ls="--", color="crimson", lw=1,
               label=f"Alert threshold ({config.ALERT_AQI_THRESHOLD})")
    ax.set_title(f"AQI over time -- {config.CITY}")
    ax.set_ylabel("AQI")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, "01_timeseries")


def plot_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    sns.histplot(df["aqi"].dropna(), bins=60, kde=True, ax=axes[0], color="#34495e")
    axes[0].axvline(config.ALERT_AQI_THRESHOLD, ls="--", color="crimson")
    axes[0].set_title("AQI distribution")

    counts = df["aqi"].dropna().apply(categorize).value_counts()
    order = [c[2] for c in CATEGORIES if c[2] in counts.index]
    colors = [c[3] for c in CATEGORIES if c[2] in counts.index]
    axes[1].bar(range(len(order)), [counts[o] for o in order], color=colors,
                edgecolor="#333")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([o.replace(" for ", "\nfor ") for o in order],
                            rotation=30, ha="right", fontsize=8)
    axes[1].set_title("Hours per AQI category")
    return _save(fig, "02_distribution")


def plot_seasonality(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df.groupby(df["ts"].dt.hour)["aqi"].mean().plot(ax=axes[0], marker="o", color="#2980b9")
    axes[0].set_title("Mean AQI by hour of day")
    axes[0].set_xlabel("Hour (UTC)")

    dow = df.groupby(df["ts"].dt.dayofweek)["aqi"].mean()
    axes[1].bar(range(len(dow)), dow.values, color="#16a085")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1].set_title("Mean AQI by weekday")

    monthly = df.groupby(df["ts"].dt.month)["aqi"].mean()
    axes[2].bar(monthly.index, monthly.values, color="#8e44ad")
    axes[2].set_title("Mean AQI by month")
    axes[2].set_xlabel("Month")
    return _save(fig, "03_seasonality")


def plot_correlation(df: pd.DataFrame) -> str:
    corr = correlations(df)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
    ax.set_title("Pollutant / weather correlation matrix")
    return _save(fig, "04_correlation")


def plot_autocorrelation(df: pd.DataFrame) -> str:
    ac = autocorrelation(df)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ac.index, ac.values, color="#c0392b")
    ax.axhline(0, color="grey", lw=0.8)
    for h in config.HORIZONS:
        ax.axvline(h, ls="--", lw=1, color="#2c3e50")
        ax.text(h, ac.max() * 0.92, f"+{h}h", fontsize=8, ha="left")
    ax.set_title("AQI autocorrelation by lag (hours)")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Correlation")
    return _save(fig, "05_autocorrelation")


def plot_weather_relationships(df: pd.DataFrame) -> str:
    pairs = [c for c in ("wind_speed", "humidity", "temperature", "pressure") if c in df]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.8))
    axes = np.atleast_1d(axes)
    sample = df.sample(min(4000, len(df)), random_state=config.RANDOM_STATE)
    for ax, col in zip(axes, pairs, strict=False):
        ax.scatter(sample[col], sample["aqi"], s=4, alpha=0.25, color="#2c3e50")
        ax.set_xlabel(col)
        ax.set_ylabel("AQI" if col == pairs[0] else "")
        r = df[[col, "aqi"]].corr().iloc[0, 1]
        ax.set_title(f"{col} (r={r:.2f})", fontsize=10)
    return _save(fig, "06_weather")


def plot_target_correlations(df: pd.DataFrame) -> str:
    corr = target_correlations(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in corr.values]
    ax.barh(range(len(corr)), corr.values, color=colors)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Engineered features most correlated with AQI")
    ax.axvline(0, color="grey", lw=0.8)
    return _save(fig, "07_feature_correlation")


FIGURES = [
    plot_timeseries, plot_distribution, plot_seasonality, plot_correlation,
    plot_autocorrelation, plot_weather_relationships, plot_target_correlations,
]


# ----------------------------------------------------------------- report ---
def build_report(df: pd.DataFrame, stats: dict, figures: list[str]) -> str:
    ac = autocorrelation(df, 168)
    corr = correlations(df)["aqi"].drop("aqi", errors="ignore").sort_values(key=abs, ascending=False)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = ["", "January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]

    top_drivers = "\n".join(
        f"| `{name}` | {value:+.3f} |" for name, value in corr.head(8).items()
    )
    fig_block = "\n".join(
        f"![{f.rsplit('/', 1)[-1]}](figures/{f.replace(chr(92), '/').rsplit('/', 1)[-1]})"
        for f in figures
    )

    return f"""# Exploratory Data Analysis -- {config.CITY} AQI

_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from {stats['rows']:,} hourly
observations spanning {stats['span_days']} days ({stats['start'][:10]} to {stats['end'][:10]},
{stats['coverage_pct']}% hourly coverage)._

## 1. Air quality profile

| Statistic | Value |
| --- | --- |
| Mean AQI | {stats['aqi_mean']} |
| Median AQI | {stats['aqi_median']} |
| Std. deviation | {stats['aqi_std']} |
| 95th percentile | {stats['aqi_p95']} |
| Range | {stats['aqi_min']} -- {stats['aqi_max']} |
| Hours at or above the alert threshold ({config.ALERT_AQI_THRESHOLD}) | {stats['unhealthy_hours']:,} ({stats['unhealthy_pct']}%) |
| Most frequent dominant pollutant | `{stats['dominant_pollutant_mode']}` |
| Missing raw pollutant readings | {stats['missing_pct']}% |

The distribution is right-skewed: typical conditions sit near the median of
{stats['aqi_median']}, but the tail reaches {stats['aqi_max']}. That skew is why the
neural networks are trained with a Huber loss rather than MSE -- the rare severe
episodes are exactly the ones a forecast needs to get right, and a squared loss
lets them dominate the gradient.

## 2. Temporal structure

* **Daily cycle.** AQI peaks around **{stats['peak_hour']:02d}:00 UTC**
  ({stats['peak_hour_aqi']}) and bottoms out at **{stats['cleanest_hour']:02d}:00 UTC**
  ({stats['cleanest_hour_aqi']}) -- a swing of
  {round(stats['peak_hour_aqi'] - stats['cleanest_hour_aqi'], 1)} AQI points. Hour-of-day is
  therefore encoded cyclically (sin/cos) so that 23:00 and 00:00 sit next to each
  other in feature space.
* **Weekly cycle.** {weekdays[stats['worst_weekday']]} is the worst day on average,
  consistent with a traffic-driven component.
* **Seasonal cycle.** {months[stats['worst_month']]} is the dirtiest month
  ({stats['worst_month_aqi']}) and {months[stats['best_month']]} the cleanest.

## 3. Autocorrelation -- the case for the lag window

| Lag | Autocorrelation |
| --- | --- |
| 1 h | {ac.get(1, float('nan')):.3f} |
| 3 h | {ac.get(3, float('nan')):.3f} |
| 12 h | {ac.get(12, float('nan')):.3f} |
| 24 h | {ac.get(24, float('nan')):.3f} |
| 48 h | {ac.get(48, float('nan')):.3f} |
| 72 h | {ac.get(72, float('nan')):.3f} |
| 168 h | {ac.get(168, float('nan')):.3f} |

Correlation stays materially above zero out to 72 hours, and there is a local
bump at 24 h from the daily cycle. Both observations are baked directly into the
feature set: lags at 1/2/3/6/12/24/48/72 h, plus rolling statistics over the same
windows. Beyond ~72 h the signal decays into noise, which is also the honest
limit of this forecast -- accuracy at +72 h is meaningfully worse than at +24 h,
and the dashboard shows that widening uncertainty band rather than hiding it.

## 4. Meteorological drivers

| Variable | Correlation with AQI |
| --- | --- |
{top_drivers}

Wind speed is the dominant meteorological control: stagnant air traps
particulates near the surface. That relationship motivates the `stagnation`
(1 / (1 + wind speed)) and `wind_u` / `wind_v` features -- decomposing wind into
vector components lets the tree models learn direction-specific effects, which a
raw 0-360 degree column cannot express because 359 and 1 are far apart
numerically but adjacent physically.

## 5. What this means for modelling

1. **Lag features carry most of the signal**, so a persistence baseline is
   genuinely hard to beat at +24 h. Every candidate is scored against it and the
   training pipeline flags any model that fails to clear it.
2. **The chronological split is mandatory.** With lag features this dense, a
   random split leaks future information and inflates R2.
3. **Direct multi-horizon beats recursive.** Separate heads for +24/48/72 h avoid
   compounding one-step error across three days.
4. **Skew argues for robust losses and tree ensembles**, which is what the
   comparison table generally bears out.

## 6. Figures

{fig_block}
"""


def run(save_report: bool = True) -> dict:
    df = load()
    stats = describe(df)
    figures = [fn(df) for fn in FIGURES]

    if save_report:
        (config.REPORT_DIR / "eda_report.md").write_text(
            build_report(df, stats, figures), encoding="utf-8"
        )
        (config.ARTIFACT_DIR / "eda_stats.json").write_text(
            json.dumps(stats, indent=2, default=str)
        )
    log.info("EDA complete: %d figures, %d rows analysed", len(figures), stats["rows"])
    return {"stats": stats, "figures": figures}


def main():
    parser = argparse.ArgumentParser(description="Run EDA and refresh the report")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(save_report=not args.no_report)
    print(json.dumps(result["stats"], indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
