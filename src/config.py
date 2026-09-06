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
# Calendar features and dashboard labels both use local time: rush-hour and
# weekday effects are local phenomena, and "AQI peaks at 13:00" is misleading
# to a Karachi reader when 13:00 UTC is 18:00 for them.
TIMEZONE = os.getenv("AQI_TIMEZONE", "Asia/Karachi")

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

# Promotion guard. Neural candidates are stochastic: a 5-seed measurement put
# their run-to-run RMSE spread at ~0.9 (experiments/exp_sequence_variance.py).
# A stochastic model must therefore beat the best deterministic model by MORE
# than that noise before it is allowed to take production, or the daily retrain
# promotes whichever seed got lucky and the forecast churns for no reason.
STOCHASTIC_NOISE_RMSE = float(os.getenv("AQI_STOCHASTIC_NOISE_RMSE", "1.0"))
STOCHASTIC_PREFIXES = ("tf_",)

# Pollutants that contribute to the EPA AQI itself.
POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]

# Auxiliary atmospheric tracers. These do NOT feed the AQI calculation -- they
# are predictors of it:
#   dust                    mineral dust; a major PM10 driver for coastal/arid
#                           Karachi, and largely independent of local traffic
#   ammonia                 precursor that forms ammonium sulfate/nitrate, i.e.
#                           secondary PM2.5 that appears hours after emission
#   aerosol_optical_depth   total column aerosol loading from satellite
#   methane                 greenhouse gas; carried as a tracer of local
#                           combustion/landfill activity. Not an AQI pollutant
#                           and expected to be weak -- included so SHAP can rank
#                           it empirically rather than assuming either way.
AUX_TRACERS = ["dust", "ammonia", "aerosol_optical_depth", "methane"]

WEATHER_COLS = [
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_direction", "precipitation",
    # Mixing depth: the vertical volume pollutants dilute into. A shallow night
    # -time boundary layer concentrates the same emissions into a fraction of
    # the air, which is the single most causal meteorological control on
    # surface concentration.
    "boundary_layer_height",
]

ALL_MEASURE_COLS = POLLUTANTS + AUX_TRACERS + WEATHER_COLS

# ---------------------------------------------------------------- alerts ----
ALERT_AQI_THRESHOLD = int(os.getenv("AQI_ALERT_THRESHOLD", "150"))  # Unhealthy
ALERT_WEBHOOK_URL = os.getenv("AQI_ALERT_WEBHOOK", "").strip()      # Slack/Discord
# Re-send an unchanged alert only after this many hours. Without a cooldown the
# hourly pipeline turns one pollution episode into dozens of identical messages.
ALERT_COOLDOWN_HOURS = int(os.getenv("AQI_ALERT_COOLDOWN_HOURS", "6"))

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
