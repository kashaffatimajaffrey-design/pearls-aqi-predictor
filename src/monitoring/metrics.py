"""Prometheus instrumentation.

Exposed on the Flask API at /metrics and scraped by the Prometheus service in
docker-compose; the Grafana dashboard in monitoring/grafana reads from there.

The metric set is deliberately small and covers the three things that actually
go wrong in a pipeline like this:

  aqi_pipeline_*    -- did the hourly/daily job run, and how long did it take?
  aqi_model_*       -- has accuracy drifted since the last training run?
  aqi_current /
  aqi_forecast_*    -- are we still producing sane predictions at all?

Metrics are also mirrored to a JSON file, because GitHub Actions runners are
ephemeral: the in-process counters die with the job, but the JSON is committed
back to the repo and re-read on the next boot so the dashboard shows continuity.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .. import config

log = logging.getLogger(__name__)

REGISTRY = CollectorRegistry()
STATE_PATH = config.ARTIFACT_DIR / "monitoring_state.json"

# ------------------------------------------------------------------ series --
PIPELINE_RUNS = Counter(
    "aqi_pipeline_runs_total", "Pipeline executions",
    ["pipeline", "status"], registry=REGISTRY,
)
PIPELINE_DURATION = Histogram(
    "aqi_pipeline_duration_seconds", "Pipeline wall-clock duration",
    ["pipeline"], registry=REGISTRY,
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800),
)
PIPELINE_LAST_SUCCESS = Gauge(
    "aqi_pipeline_last_success_timestamp", "Unix time of the last successful run",
    ["pipeline"], registry=REGISTRY,
)

MODEL_RMSE = Gauge("aqi_model_rmse", "Hold-out RMSE of the production model",
                   ["model"], registry=REGISTRY)
MODEL_MAE = Gauge("aqi_model_mae", "Hold-out MAE of the production model",
                  ["model"], registry=REGISTRY)
MODEL_R2 = Gauge("aqi_model_r2", "Hold-out R2 of the production model",
                 ["model"], registry=REGISTRY)
MODEL_RMSE_HORIZON = Gauge("aqi_model_rmse_by_horizon", "Hold-out RMSE per horizon",
                           ["horizon"], registry=REGISTRY)

CURRENT_AQI = Gauge("aqi_current", "Latest observed AQI", ["city"], registry=REGISTRY)
FORECAST_AQI = Gauge("aqi_forecast", "Forecast AQI", ["city", "horizon"], registry=REGISTRY)
PREDICTIONS = Counter("aqi_predictions_total", "Forecasts served", registry=REGISTRY)
ALERTS = Counter("aqi_alerts_total", "Hazardous-AQI alerts raised",
                 ["category"], registry=REGISTRY)
FEATURE_FRESHNESS = Gauge(
    "aqi_feature_age_seconds", "Age of the newest row in the feature store",
    registry=REGISTRY,
)

API_REQUESTS = Counter("aqi_api_requests_total", "Flask API requests",
                       ["endpoint", "status"], registry=REGISTRY)
API_LATENCY = Histogram("aqi_api_latency_seconds", "Flask API latency",
                        ["endpoint"], registry=REGISTRY)


# ------------------------------------------------------------------- state --
def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save_state(patch: dict) -> None:
    state = _load_state()
    state.update(patch)
    state["updated_at"] = datetime.now(UTC).isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def get_state() -> dict:
    return _load_state()


# ------------------------------------------------------------- recorders ----
def record_pipeline_run(pipeline: str, duration_s: float, ok: bool = True) -> None:
    PIPELINE_RUNS.labels(pipeline=pipeline, status="success" if ok else "failure").inc()
    PIPELINE_DURATION.labels(pipeline=pipeline).observe(duration_s)
    if ok:
        now = datetime.now(UTC)
        PIPELINE_LAST_SUCCESS.labels(pipeline=pipeline).set(now.timestamp())
        _save_state({f"{pipeline}_last_success": now.isoformat(),
                     f"{pipeline}_last_duration_s": duration_s})


def record_model_metrics(model_name: str, metrics: dict) -> None:
    MODEL_RMSE.labels(model=model_name).set(metrics.get("rmse", 0.0))
    MODEL_MAE.labels(model=model_name).set(metrics.get("mae", 0.0))
    MODEL_R2.labels(model=model_name).set(metrics.get("r2", 0.0))
    for horizon, values in (metrics.get("per_horizon") or {}).items():
        MODEL_RMSE_HORIZON.labels(horizon=horizon).set(values.get("rmse", 0.0))
    _save_state({
        "model_name": model_name,
        "model_rmse": metrics.get("rmse"),
        "model_mae": metrics.get("mae"),
        "model_r2": metrics.get("r2"),
    })


def record_prediction(result: dict) -> None:
    city = result.get("city", config.CITY)
    current = (result.get("current") or {}).get("aqi")
    if current is not None:
        CURRENT_AQI.labels(city=city).set(current)
    for f in result.get("forecast") or []:
        FORECAST_AQI.labels(city=city, horizon=f"{f['horizon_hours']}h").set(f["aqi"])
    for a in result.get("alerts") or []:
        ALERTS.labels(category=a.get("category", "unknown")).inc()
    PREDICTIONS.inc()

    as_of = result.get("as_of")
    if as_of:
        try:
            age = (datetime.now(UTC) - datetime.fromisoformat(as_of)).total_seconds()
            FEATURE_FRESHNESS.set(max(0.0, age))
        except ValueError:
            pass


def record_api_request(endpoint: str, status: int, latency_s: float) -> None:
    API_REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()
    API_LATENCY.labels(endpoint=endpoint).observe(latency_s)


def hydrate_from_state() -> None:
    """Re-seed gauges from the committed JSON so a fresh process does not report
    zeros for metrics that were last written by an ephemeral CI runner."""
    state = _load_state()
    if state.get("model_name"):
        MODEL_RMSE.labels(model=state["model_name"]).set(state.get("model_rmse") or 0.0)
        MODEL_MAE.labels(model=state["model_name"]).set(state.get("model_mae") or 0.0)
        MODEL_R2.labels(model=state["model_name"]).set(state.get("model_r2") or 0.0)
    for pipeline in ("feature", "training"):
        stamp = state.get(f"{pipeline}_last_success")
        if stamp:
            try:
                PIPELINE_LAST_SUCCESS.labels(pipeline=pipeline).set(
                    datetime.fromisoformat(stamp).timestamp()
                )
            except ValueError:
                continue


def render_metrics() -> bytes:
    """Prometheus text exposition format."""
    hydrate_from_state()
    return generate_latest(REGISTRY)
