"""Raw data connectors.

Two production sources are implemented behind one contract:

    fetch(lat, lon, start, end) -> DataFrame[ts, pm2_5, pm10, no2, so2, o3, co,
                                             temperature, humidity, pressure,
                                             wind_speed, wind_direction,
                                             precipitation]

* OpenWeather -- the assignment's named provider. Air Pollution History gives
  hourly pollutants back to Nov 2020.
* Open-Meteo  -- keyless fallback so the pipeline runs with zero credentials
  (CI, graders, a fresh clone). Hourly pollutants + ERA5 weather reanalysis.
* AQICN       -- current observation only (no history); used as a live
  cross-check on the dashboard rather than for training.

All timestamps are tz-aware UTC, floored to the hour.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .. import config
from .validation import validate

log = logging.getLogger(__name__)

TIMEOUT = 45
NUMERIC_COLS = config.ALL_MEASURE_COLS


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts"] + NUMERIC_COLS)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes and columns so every source returns the same shape."""
    if df.empty:
        return _empty()
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.floor("h")
    for c in NUMERIC_COLS:
        if c not in df:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["ts"] + NUMERIC_COLS]

    # Gate every source through the same plausibility checks, so a broken
    # upstream reading cannot reach feature engineering or the training set.
    df, report = validate(df)
    if report.get("warnings"):
        log.warning("data validation warnings: %s", report["warnings"])

    df = df.dropna(subset=["pm2_5"])
    return df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------- Open-Meteo --
_OM_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
_OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_OM_FORECAST = "https://api.open-meteo.com/v1/forecast"

_OM_AQ_VARS = ("pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,"
               "dust,ammonia,aerosol_optical_depth,methane")
_OM_WX_VARS = (
    "temperature_2m,relative_humidity_2m,surface_pressure,"
    "wind_speed_10m,wind_direction_10m,precipitation,boundary_layer_height"
)

_OM_AQ_RENAME = {
    "pm10": "pm10",
    "pm2_5": "pm2_5",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
    "dust": "dust",
    "ammonia": "ammonia",
    "aerosol_optical_depth": "aerosol_optical_depth",
    "methane": "methane",
}
_OM_WX_RENAME = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "precipitation": "precipitation",
    "boundary_layer_height": "boundary_layer_height",
}

# Open-Meteo serves the forecast endpoints (with past_days) fresher than the
# dated archive, so recent windows go through those.
_RECENT_DAYS = 80


def _om_frame(payload: dict, rename: dict) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        return pd.DataFrame(columns=["ts"])
    df = pd.DataFrame(hourly).rename(columns={"time": "ts", **rename})
    keep = ["ts"] + [v for v in rename.values() if v in df.columns]
    return df[keep]


def _om_get(url: str, params: dict) -> pd.DataFrame:
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _om_params(lat, lon, start, end, hourly):
    now = datetime.now(timezone.utc)
    if (now - start) <= timedelta(days=_RECENT_DAYS):
        past_days = max(1, min(92, (now - start).days + 1))
        return {
            "latitude": lat, "longitude": lon, "hourly": hourly,
            "past_days": past_days, "forecast_days": 1, "timezone": "UTC",
        }, True
    return {
        "latitude": lat, "longitude": lon, "hourly": hourly,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(), "timezone": "UTC",
    }, False


def _openmeteo_weather(lat, lon, start, end) -> pd.DataFrame:
    params, recent = _om_params(lat, lon, start, end, _OM_WX_VARS)
    url = _OM_FORECAST if recent else _OM_ARCHIVE
    df = _om_frame(_om_get(url, params), _OM_WX_RENAME)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _openmeteo(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    aq_params, _ = _om_params(lat, lon, start, end, _OM_AQ_VARS)
    aq_df = _om_frame(_om_get(_OM_AQ, aq_params), _OM_AQ_RENAME)
    if aq_df.empty:
        return _empty()
    aq_df["ts"] = pd.to_datetime(aq_df["ts"], utc=True)

    try:
        wx_df = _openmeteo_weather(lat, lon, start, end)
    except Exception as exc:  # pragma: no cover - network best effort
        log.warning("weather fetch failed: %s", exc)
        wx_df = pd.DataFrame(columns=["ts"])

    merged = aq_df.merge(wx_df, on="ts", how="left") if not wx_df.empty else aq_df
    merged = merged[(merged["ts"] >= start) & (merged["ts"] <= end)]
    return _finalize(merged)


# -------------------------------------------------------------- OpenWeather --
_OW_POLLUTION = "https://api.openweathermap.org/data/2.5/air_pollution/history"

_OW_RENAME = {"pm2_5": "pm2_5", "pm10": "pm10", "no2": "no2",
              "so2": "so2", "o3": "o3", "co": "co"}


def _openweather(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    key = config.OPENWEATHER_API_KEY
    if not key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set")

    rows: list[dict] = []
    cursor = start
    while cursor < end:  # the history endpoint caps each call, so walk in chunks
        chunk_end = min(cursor + timedelta(days=7), end)
        r = requests.get(_OW_POLLUTION, timeout=TIMEOUT, params={
            "lat": lat, "lon": lon,
            "start": int(cursor.timestamp()), "end": int(chunk_end.timestamp()),
            "appid": key,
        })
        r.raise_for_status()
        for item in r.json().get("list", []):
            comp = item.get("components", {})
            rows.append({
                "ts": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
                **{dst: comp.get(src) for src, dst in _OW_RENAME.items()},
            })
        cursor = chunk_end

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty()

    # OpenWeather's free tier has no bulk weather history, so meteorology comes
    # from Open-Meteo's ERA5 reanalysis on the same lat/lon.
    try:
        wx = _openmeteo_weather(lat, lon, start, end)
        if not wx.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.floor("h")
            df = df.merge(wx, on="ts", how="left")
    except Exception as exc:  # pragma: no cover
        log.warning("weather enrichment failed: %s", exc)

    return _finalize(df)


# -------------------------------------------------------------------- AQICN --
def fetch_aqicn_current(lat: float | None = None, lon: float | None = None) -> dict | None:
    """Current reading from the nearest AQICN station, or None if unavailable."""
    if not config.AQICN_TOKEN:
        return None
    lat = config.LAT if lat is None else lat
    lon = config.LON if lon is None else lon
    try:
        r = requests.get(
            f"https://api.waqi.info/feed/geo:{lat};{lon}/",
            params={"token": config.AQICN_TOKEN}, timeout=TIMEOUT,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") != "ok":
            return None
        d = payload["data"]
        return {
            "aqi": d.get("aqi"),
            "station": (d.get("city") or {}).get("name"),
            "dominant_pollutant": d.get("dominentpol"),
            "observed_at": (d.get("time") or {}).get("iso"),
        }
    except Exception as exc:  # pragma: no cover
        log.warning("AQICN lookup failed: %s", exc)
        return None


# ------------------------------------------------------------------ router --
SOURCES = {"openmeteo": _openmeteo, "openweather": _openweather}


def _as_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def fetch_range(start, end, source: str | None = None,
                lat: float | None = None, lon: float | None = None) -> pd.DataFrame:
    """Fetch raw observations for [start, end].

    Falls back to Open-Meteo when the preferred source errors, so a missing or
    expired API key degrades the pipeline instead of breaking it.
    """
    source = source or config.resolved_source()
    lat = config.LAT if lat is None else lat
    lon = config.LON if lon is None else lon
    start_dt = _as_utc(start).to_pydatetime()
    end_dt = _as_utc(end).to_pydatetime()

    fn = SOURCES.get(source, _openmeteo)
    try:
        df = fn(lat, lon, start_dt, end_dt)
        if not df.empty:
            log.info("fetched %d rows from %s", len(df), source)
            return df
        log.warning("%s returned no rows", source)
    except Exception as exc:
        log.warning("source %s failed (%s)", source, exc)

    if source != "openmeteo":
        log.info("falling back to open-meteo")
        return _openmeteo(lat, lon, start_dt, end_dt)
    return _empty()


def fetch_recent(hours: int = 168, **kw) -> pd.DataFrame:
    """Trailing window fetch. The 7-day default is enough context to rebuild
    every lag and rolling feature on an hourly run."""
    end = datetime.now(timezone.utc)
    return fetch_range(end - timedelta(hours=hours), end, **kw)
