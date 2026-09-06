"""Central configuration. Everything is env-overridable so the same code runs
locally, in GitHub Actions, on Streamlit Cloud and inside Docker."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("AQI_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"
ARTIFACT_DIR = DATA_DIR / "artifacts"
MODEL_DIR = Path(os.getenv("AQI_MODEL_DIR", ROOT / "models"))
REGISTRY_DIR = MODEL_DIR / "registry"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

for _d in (RAW_DIR, FEATURE_DIR, ARTIFACT_DIR, REGISTRY_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- location --
CITY = os.getenv("AQI_CITY", "Karachi")
COUNTRY = os.getenv("AQI_COUNTRY", "PK")
LAT = float(os.getenv("AQI_LAT", "24.8607"))
LON = float(os.getenv("AQI_LON", "67.0011"))
TIMEZONE = os.getenv("AQI_TIMEZONE", "UTC")

# ------------------------------------------------------------------ source --
# "auto"        -> OpenWeather if a key is present, else Open-Meteo
# "openweather" -> force OpenWeather (needs OPENWEATHER_API_KEY)
# "openmeteo"   -> force Open-Meteo (keyless, works out of the box)
# "aqicn"       -> AQICN current observation (needs AQICN_TOKEN, no history)
DATA_SOURCE = os.getenv("AQI_DATA_SOURCE", "auto").lower()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "").strip()

# --------------------------------------------------------- feature store ----
# "auto" -> Hopsworks when HOPSWORKS_API_KEY is set, else local parquet store
FEATURE_STORE = os.getenv("AQI_FEATURE_STORE", "auto").lower()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "").strip()
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "").strip()
FG_NAME = os.getenv("AQI_FG_NAME", "aqi_features")
FG_VERSION = int(os.getenv("AQI_FG_VERSION", "1"))
MODEL_REGISTRY_NAME = os.getenv("AQI_MODEL_NAME", "aqi_forecaster")

# ------------------------------------------------------------- modelling ----
HORIZONS = [24, 48, 72]           # hours ahead -> "next 3 days"
TARGET_COLS = [f"aqi_t_plus_{h}h" for h in HORIZONS]
BACKFILL_DAYS = int(os.getenv("AQI_BACKFILL_DAYS", "365"))
TEST_SIZE_HOURS = int(os.getenv("AQI_TEST_HOURS", str(24 * 21)))  # last 3 weeks held out
RANDOM_STATE = 42

POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]
WEATHER_COLS = [
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_direction", "precipitation",
]

# ---------------------------------------------------------------- alerts ----
ALERT_AQI_THRESHOLD = int(os.getenv("AQI_ALERT_THRESHOLD", "150"))  # Unhealthy
ALERT_WEBHOOK_URL = os.getenv("AQI_ALERT_WEBHOOK", "").strip()      # Slack/Discord

# ------------------------------------------------------------ monitoring ----
METRICS_PORT = int(os.getenv("AQI_METRICS_PORT", "8000"))
API_PORT = int(os.getenv("AQI_API_PORT", "5000"))


def summary() -> dict:
    return {
        "city": CITY, "country": COUNTRY, "lat": LAT, "lon": LON,
        "data_source": resolved_source(),
        "feature_store": resolved_store(),
        "horizons_h": HORIZONS,
        "alert_threshold": ALERT_AQI_THRESHOLD,
    }


def resolved_source() -> str:
    if DATA_SOURCE != "auto":
        return DATA_SOURCE
    return "openweather" if OPENWEATHER_API_KEY else "openmeteo"


def resolved_store() -> str:
    if FEATURE_STORE != "auto":
        return FEATURE_STORE
    return "hopsworks" if (HOPSWORKS_API_KEY and HOPSWORKS_PROJECT) else "local"
