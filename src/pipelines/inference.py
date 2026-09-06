"""Inference: produce the next-3-day AQI forecast.

Loads the production bundle from the Model Registry and the most recent row
from the Feature Store, then emits one prediction per horizon along with a
category, health advice and an uncertainty band.

The band comes from the held-out test residuals recorded at training time
(+/- 1.28 * RMSE ~ an 80% interval under a normal residual assumption). It is a
calibrated-ish range rather than a formal prediction interval, and it is
labelled as such on the dashboard.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config
from ..alerts import evaluate_forecast_alerts
from ..aqi import advice, categorize, category_color
from ..data import fetch_recent, fetch_weather_forecast
from ..features import add_future_weather, latest_feature_row
from ..models import load_bundle
from ..monitoring.metrics import record_prediction
from ..store import get_feature_store, get_model_registry

log = logging.getLogger(__name__)

Z80 = 1.2816  # z-score for an 80% two-sided interval


@lru_cache(maxsize=1)
def _load_production():
    registry = get_model_registry()
    path = registry.production_path()
    if path is None:
        raise RuntimeError(
            "no model in the registry -- run `python -m src.pipelines.training_pipeline`"
        )
    return load_bundle(path)


def clear_cache() -> None:
    """Drop the cached model so a freshly trained version is picked up."""
    _load_production.cache_clear()


def _latest_features(live: bool) -> pd.DataFrame:
    """Newest feature row, preferring the Feature Store and falling back to a
    live API pull when the store is stale or empty."""
    if not live:
        store_df = get_feature_store().read()
        if not store_df.empty:
            age = datetime.now(UTC) - pd.Timestamp(store_df["ts"].max()).to_pydatetime()
            if age <= timedelta(hours=6):
                return store_df.tail(1).reset_index(drop=True)
            log.info("feature store is %.1fh stale; pulling live data", age.total_seconds() / 3600)

    raw = fetch_recent(hours=168)
    if raw.empty:
        store_df = get_feature_store().read()
        if store_df.empty:
            raise RuntimeError("no features available from the store or the API")
        return store_df.tail(1).reset_index(drop=True)
    return latest_feature_row(raw)


def predict(live: bool = False) -> dict:
    pipeline, metadata = _load_production()
    feat_cols = metadata.get("feature_columns") or []

    row = _latest_features(live=live)
    if row.empty:
        raise RuntimeError("could not assemble a feature row for inference")

    # Known-future weather from a real forward forecast. If the model was not
    # trained with these columns they are simply ignored by the feature
    # selection below, so this is safe for older bundles too.
    if metadata.get("uses_future_weather") or any("_fut_" in c for c in feat_cols):
        forecast_wx = fetch_weather_forecast()
        if forecast_wx.empty:
            log.warning("no weather forecast available; future-weather features "
                        "will be imputed, degrading accuracy")
        row = add_future_weather(row, forecast=forecast_wx)

    missing = [c for c in feat_cols if c not in row.columns]
    for c in missing:
        row[c] = np.nan
    if missing:
        log.warning("%d feature(s) missing at inference; imputed: %s",
                    len(missing), missing[:5])

    X = row[feat_cols]
    y_pred = np.asarray(pipeline.predict(X), dtype=float).ravel()
    y_pred = np.clip(y_pred, 0, 500)

    as_of = pd.Timestamp(row["ts"].iloc[0])
    current_aqi = float(row["aqi"].iloc[0])
    per_horizon = metadata.get("metrics_per_horizon") or {}

    forecasts = []
    for i, horizon in enumerate(config.HORIZONS):
        value = float(y_pred[i])
        rmse = float(per_horizon.get(f"{horizon}h", {}).get("rmse", 0.0))
        margin = Z80 * rmse
        forecasts.append({
            "horizon_hours": horizon,
            "horizon_days": horizon // 24,
            "valid_at": (as_of + timedelta(hours=horizon)).isoformat(),
            "aqi": round(value, 1),
            "aqi_lower": round(max(0.0, value - margin), 1),
            "aqi_upper": round(min(500.0, value + margin), 1),
            "interval": "80%",
            "category": categorize(value),
            "color": category_color(value),
            "advice": advice(value),
            "model_rmse": round(rmse, 2),
        })

    alerts = evaluate_forecast_alerts(forecasts)

    result = {
        "city": config.CITY,
        "country": config.COUNTRY,
        "coordinates": {"lat": config.LAT, "lon": config.LON},
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of": as_of.isoformat(),
        "current": {
            "aqi": round(current_aqi, 1),
            "category": categorize(current_aqi),
            "color": category_color(current_aqi),
            "dominant_pollutant": str(row.get("dominant_pollutant", pd.Series(["n/a"])).iloc[0]),
            "pm2_5": _safe(row, "pm2_5"),
            "pm10": _safe(row, "pm10"),
            "temperature": _safe(row, "temperature"),
            "humidity": _safe(row, "humidity"),
            "wind_speed": _safe(row, "wind_speed"),
        },
        "forecast": forecasts,
        "model": {
            "name": metadata.get("best_model"),
            "version": metadata.get("version"),
            "trained_at": metadata.get("trained_at"),
            "rmse": metadata.get("metrics", {}).get("rmse"),
            "mae": metadata.get("metrics", {}).get("mae"),
            "r2": metadata.get("metrics", {}).get("r2"),
        },
        "alerts": alerts,
        "data_source": config.resolved_source(),
    }

    (config.ARTIFACT_DIR / "latest_forecast.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    record_prediction(result)
    return result


def _safe(row: pd.DataFrame, col: str):
    if col not in row.columns:
        return None
    value = row[col].iloc[0]
    return None if pd.isna(value) else round(float(value), 2)


def backtest(days: int = 30) -> pd.DataFrame:
    """Replay the production model over recent history so the dashboard can show
    predicted-vs-actual rather than only forward-looking numbers."""
    pipeline, metadata = _load_production()
    feat_cols = metadata.get("feature_columns") or []

    store_df = get_feature_store().read()
    if store_df.empty:
        return pd.DataFrame()

    window = store_df.tail(days * 24 + max(config.HORIZONS)).copy()
    for c in feat_cols:
        if c not in window.columns:
            window[c] = np.nan

    preds = np.clip(np.asarray(pipeline.predict(window[feat_cols]), dtype=float), 0, 500)

    frames = []
    actual = window.set_index("ts")["aqi"]
    for i, horizon in enumerate(config.HORIZONS):
        valid_at = window["ts"] + pd.Timedelta(hours=horizon)
        frames.append(pd.DataFrame({
            "issued_at": window["ts"].to_numpy(),
            "valid_at": valid_at.to_numpy(),
            "horizon_hours": horizon,
            "predicted": preds[:, i],
            "actual": valid_at.map(actual).to_numpy(),
        }))

    out = pd.concat(frames, ignore_index=True).dropna(subset=["actual"])
    out["error"] = out["predicted"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    return out.sort_values(["horizon_hours", "valid_at"]).reset_index(drop=True)


def main() -> dict:
    parser = argparse.ArgumentParser(description="Generate the 3-day AQI forecast")
    parser.add_argument("--live", action="store_true",
                        help="bypass the feature store and pull fresh data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = predict(live=args.live)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
