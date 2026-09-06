"""Feature engineering.

Turns the raw hourly observation table into the modelling matrix. Three groups:

1. Calendar   -- hour/day/month, cyclically encoded, plus a weekend flag.
2. Lag        -- AQI and pollutant values at t-1 ... t-72h.
3. Derived    -- rolling means/stds, AQI change rate (the assignment calls this
                 out explicitly), wind decomposed into u/v components, and a
                 ventilation proxy.

Every feature is computed only from information available at time t, so the
matrix can be reused unchanged at inference time. The targets are AQI at t+24h,
t+48h and t+72h -- direct multi-horizon forecasting, which avoids the error
compounding of a recursive one-step model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..aqi import compute_aqi_frame

FEATURE_VERSION = "v1"

LAGS = [1, 2, 3, 6, 12, 24, 48, 72]
ROLL_WINDOWS = [6, 12, 24, 72]
# Lagged because each is causally upstream of surface AQI: dust blows in,
# ammonia converts to secondary aerosol over hours, and the mixing depth
# earlier in the day sets how concentrated the air already is.
LAG_BASE = ["aqi", "pm2_5", "pm10", "no2", "o3", "dust", "boundary_layer_height"]

# Weather variables genuinely knowable in advance from a forecast. These are the
# only features allowed to reference a time AFTER t -- everything else is
# strictly causal. Measured worth: ~8.9% RMSE (experiments/exp_future_weather).
FUTURE_WX_COLS = [
    "temperature", "humidity", "pressure", "wind_speed",
    "precipitation", "boundary_layer_height",
]


def _cyclical(df: pd.DataFrame, col: str, period: int) -> None:
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)


def build_features(raw: pd.DataFrame, *, dropna: bool = True) -> pd.DataFrame:
    """Raw observations -> feature matrix (one row per hour).

    `dropna=False` keeps the warm-up rows, which matters at inference time where
    we only care about the final row and cannot afford to lose it.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)

    # Reindex onto a gapless hourly grid; short API gaps are interpolated so the
    # lag structure stays aligned to real wall-clock offsets.
    full_index = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h", tz="UTC")
    df = df.set_index("ts").reindex(full_index)
    df.index.name = "ts"
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].interpolate(limit=6, limit_direction="both")
    df = df.reset_index()

    df = compute_aqi_frame(df)

    # ---------------------------------------------------------- calendar ----
    ts_local = df["ts"].dt.tz_convert(config.TIMEZONE)
    df["hour"] = ts_local.dt.hour
    df["day_of_week"] = ts_local.dt.dayofweek
    df["day_of_year"] = ts_local.dt.dayofyear
    df["month"] = ts_local.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    _cyclical(df, "hour", 24)
    _cyclical(df, "day_of_week", 7)
    _cyclical(df, "month", 12)

    # ------------------------------------------------------------- lags -----
    for col in LAG_BASE:
        if col not in df:
            continue
        for lag in LAGS:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    # ---------------------------------------------------------- rolling -----
    for window in ROLL_WINDOWS:
        roll = df["aqi"].shift(1).rolling(window, min_periods=max(2, window // 3))
        df[f"aqi_roll_mean_{window}h"] = roll.mean()
        df[f"aqi_roll_std_{window}h"] = roll.std()
        df[f"aqi_roll_max_{window}h"] = roll.max()
        df[f"aqi_roll_min_{window}h"] = roll.min()
        for col in ("pm2_5", "pm10"):
            if col in df:
                df[f"{col}_roll_mean_{window}h"] = (
                    df[col].shift(1).rolling(window, min_periods=max(2, window // 3)).mean()
                )

    # ---------------------------------------------------- change / trend ----
    # "AQI change rate" from the brief, at several resolutions.
    df["aqi_change_1h"] = df["aqi"].diff(1)
    df["aqi_change_3h"] = df["aqi"].diff(3)
    df["aqi_change_24h"] = df["aqi"].diff(24)
    df["aqi_change_rate_24h"] = df["aqi_change_24h"] / 24.0
    df["aqi_pct_change_24h"] = df["aqi"].pct_change(24).replace([np.inf, -np.inf], np.nan)
    df["aqi_accel"] = df["aqi_change_1h"].diff(1)
    df["aqi_vs_24h_mean"] = df["aqi"] - df["aqi_roll_mean_24h"]
    df["pm_ratio"] = (df["pm2_5"] / df["pm10"].replace(0, np.nan)).clip(0, 1.5)

    # ----------------------------------------------------------- weather ----
    if "wind_direction" in df:
        rad = np.deg2rad(df["wind_direction"].astype(float))
        speed = df.get("wind_speed", pd.Series(0.0, index=df.index)).astype(float)
        df["wind_u"] = -speed * np.sin(rad)
        df["wind_v"] = -speed * np.cos(rad)
    if "wind_speed" in df:
        # Stagnant air traps particulates; this is the strongest weather signal.
        df["wind_speed_roll_24h"] = df["wind_speed"].rolling(24, min_periods=6).mean()
        df["stagnation"] = 1.0 / (1.0 + df["wind_speed"].astype(float))
    if "boundary_layer_height" in df:
        blh = df["boundary_layer_height"].astype(float)
        df["blh_roll_mean_24h"] = blh.rolling(24, min_periods=6).mean()
        df["blh_min_24h"] = blh.rolling(24, min_periods=6).min()
        # Ventilation index = mixing depth x wind speed: the standard measure of
        # how much air is actually available to dilute emissions. Low values are
        # what turn ordinary emissions into a pollution episode.
        if "wind_speed" in df:
            df["ventilation_index"] = blh * df["wind_speed"].astype(float)
            df["ventilation_roll_24h"] = df["ventilation_index"].rolling(
                24, min_periods=6
            ).mean()

    if "dust" in df:
        df["dust_change_24h"] = df["dust"].diff(24)

    if {"temperature", "humidity"}.issubset(df.columns):
        df["temp_humidity"] = df["temperature"] * df["humidity"] / 100.0
        df["temp_range_24h"] = (
            df["temperature"].rolling(24, min_periods=6).max()
            - df["temperature"].rolling(24, min_periods=6).min()
        )

    df["city"] = config.CITY
    df["feature_version"] = FEATURE_VERSION

    if dropna:
        required = [c for c in df.columns if c.endswith("_lag_72h")] + ["aqi"]
        df = df.dropna(subset=required).reset_index(drop=True)

    return df


def add_future_weather(df: pd.DataFrame, forecast: pd.DataFrame | None = None,
                       horizons: list[int] | None = None) -> pd.DataFrame:
    """Attach weather at t+24/48/72h as known-future covariates.

    Two sources, and the distinction matters:

    * **Training** (`forecast=None`) shifts *observed* weather backwards. That is
      perfect foresight -- the standard "perfect prog" setup. It is an optimistic
      upper bound, not what production will achieve.
    * **Inference** (`forecast=` a real forward forecast) uses predicted weather,
      which carries error. This is the honest path and the one that runs live.

    The gap between the two is a known train/serve mismatch, documented rather
    than hidden: the model is trained on better weather information than it gets
    at serving time, so realised accuracy sits below the measured ceiling.
    """
    horizons = horizons or config.HORIZONS
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)

    if forecast is not None and not forecast.empty:
        fc = forecast.copy()
        fc["ts"] = pd.to_datetime(fc["ts"], utc=True)
        fc = fc.set_index("ts")
        for h in horizons:
            target_ts = out["ts"] + pd.Timedelta(hours=h)
            for col in FUTURE_WX_COLS:
                if col in fc.columns:
                    out[f"{col}_fut_{h}h"] = target_ts.map(fc[col]).to_numpy()
    else:
        for h in horizons:
            for col in FUTURE_WX_COLS:
                if col in out.columns:
                    out[f"{col}_fut_{h}h"] = out[col].shift(-h)

    # Ventilation at the forecast hour -- the same physics that makes the
    # present-time ventilation index useful, projected forward.
    for h in horizons:
        blh, wind = f"boundary_layer_height_fut_{h}h", f"wind_speed_fut_{h}h"
        if blh in out.columns and wind in out.columns:
            out[f"ventilation_fut_{h}h"] = out[blh] * out[wind]
    return out


def build_targets(features: pd.DataFrame) -> pd.DataFrame:
    """Attach t+24h / t+48h / t+72h AQI targets (shifted backwards in time)."""
    df = features.copy()
    for horizon, col in zip(config.HORIZONS, config.TARGET_COLS, strict=False):
        df[col] = df["aqi"].shift(-horizon)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: every numeric column that is not a target or an identifier."""
    exclude = set(config.TARGET_COLS) | {
        "ts", "city", "feature_version", "dominant_pollutant",
        "aqi_pm2_5", "aqi_pm10", "aqi_no2", "aqi_so2", "aqi_o3", "aqi_co",
    }
    # Drop columns with no observed values at all -- a provider may simply not
    # cover a variable at this location (CAMS has no ammonia for Karachi), and
    # an empty column is noise that only creates alignment hazards downstream.
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not df[c].isna().all()
    ]


def latest_feature_row(raw: pd.DataFrame) -> pd.DataFrame:
    """The single most recent complete feature row -- the inference input."""
    feats = build_features(raw, dropna=False)
    if feats.empty:
        return feats
    numeric = feats.select_dtypes(include=[np.number]).columns
    feats[numeric] = feats[numeric].ffill()
    return feats.tail(1).reset_index(drop=True)
