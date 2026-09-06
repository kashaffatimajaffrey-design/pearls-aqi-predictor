"""Flask REST API.

Serves the forecast as JSON so the prediction is consumable by something other
than the dashboard, and exposes the Prometheus scrape endpoint.

    GET /                 service metadata
    GET /health           liveness + data/model freshness
    GET /predict          3-day forecast (?live=1 to bypass the feature store)
    GET /current          latest observation (+ AQICN cross-check)
    GET /history?hours=   recent observed AQI series
    GET /backtest?days=   predicted vs actual replay
    GET /models           model comparison table from the last training run
    GET /explain?horizon= SHAP explanation for a forecast horizon
    GET /alerts           active alerts
    GET /metrics          Prometheus exposition

Run it with:  python -m api.flask_api      (or gunicorn/waitress in production)
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.alerts import evaluate_forecast_alerts  # noqa: E402
from src.data import fetch_aqicn_current  # noqa: E402
from src.explain import explain_prediction, load_shap_summary  # noqa: E402
from src.monitoring.metrics import record_api_request, render_metrics  # noqa: E402
from src.pipelines import inference  # noqa: E402
from src.store import get_feature_store, get_model_registry  # noqa: E402

log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


# ------------------------------------------------------------ middleware ----
@app.before_request
def _start_timer():
    request._started = time.perf_counter()


@app.after_request
def _record(response):
    started = getattr(request, "_started", None)
    if started is not None:
        record_api_request(
            request.endpoint or request.path,
            response.status_code,
            time.perf_counter() - started,
        )
    return response


@app.errorhandler(Exception)
def _handle(exc):
    log.exception("unhandled API error")
    return jsonify({"error": type(exc).__name__, "detail": str(exc)}), 500


# ----------------------------------------------------------------- routes ---
@app.get("/")
def index():
    return jsonify({
        "service": "Pearls AQI Predictor API",
        "version": "1.0.0",
        "city": config.CITY,
        "horizons_hours": config.HORIZONS,
        "config": config.summary(),
        "endpoints": [
            "/health", "/predict", "/current", "/history", "/backtest",
            "/models", "/explain", "/alerts", "/metrics",
        ],
    })


@app.get("/health")
def health():
    """Liveness plus the two things that actually indicate trouble: stale
    features and a missing model."""
    checks: dict[str, object] = {}
    status = "healthy"

    try:
        df = get_feature_store().read()
        if df.empty:
            checks["feature_store"] = {"ok": False, "detail": "empty"}
            status = "degraded"
        else:
            age_h = (
                datetime.now(UTC) - df["ts"].max().to_pydatetime()
            ).total_seconds() / 3600
            ok = age_h < 6
            checks["feature_store"] = {
                "ok": ok, "rows": int(len(df)),
                "latest": str(df["ts"].max()), "age_hours": round(age_h, 2),
            }
            if not ok:
                status = "degraded"
    except Exception as exc:
        checks["feature_store"] = {"ok": False, "detail": str(exc)}
        status = "unhealthy"

    try:
        path = get_model_registry().production_path()
        checks["model_registry"] = {"ok": path is not None, "path": str(path)}
        if path is None:
            status = "degraded"
    except Exception as exc:
        checks["model_registry"] = {"ok": False, "detail": str(exc)}
        status = "unhealthy"

    code = 200 if status != "unhealthy" else 503
    return jsonify({
        "status": status,
        "checked_at": datetime.now(UTC).isoformat(),
        "checks": checks,
    }), code


@app.get("/predict")
def predict():
    live = request.args.get("live", "").lower() in ("1", "true", "yes")
    return jsonify(inference.predict(live=live))


@app.get("/current")
def current():
    result = inference.predict()
    payload = {"as_of": result["as_of"], **result["current"]}
    cross_check = fetch_aqicn_current()
    if cross_check:
        payload["aqicn_reference"] = cross_check
    return jsonify(payload)


@app.get("/history")
def history():
    hours = min(int(request.args.get("hours", 168)), 24 * 365)
    df = get_feature_store().read()
    if df.empty:
        return jsonify({"rows": 0, "data": []})
    tail = df.tail(hours)
    cols = [c for c in ("ts", "aqi", "pm2_5", "pm10", "no2", "o3",
                        "temperature", "humidity", "wind_speed") if c in tail.columns]
    records = tail[cols].copy()
    records["ts"] = records["ts"].astype(str)
    return jsonify({"rows": len(records), "city": config.CITY,
                    "data": records.to_dict(orient="records")})


@app.get("/backtest")
def backtest():
    days = min(int(request.args.get("days", 14)), 120)
    df = inference.backtest(days=days)
    if df.empty:
        return jsonify({"rows": 0, "data": [], "summary": {}})

    summary = {
        f"{h}h": {
            "mae": round(float(g["abs_error"].mean()), 2),
            "rmse": round(float((g["error"] ** 2).mean() ** 0.5), 2),
            "n": int(len(g)),
        }
        for h, g in df.groupby("horizon_hours")
    }
    out = df.copy()
    out["issued_at"] = out["issued_at"].astype(str)
    out["valid_at"] = out["valid_at"].astype(str)
    return jsonify({"rows": len(out), "summary": summary,
                    "data": out.to_dict(orient="records")})


@app.get("/models")
def models():
    import json

    path = config.ARTIFACT_DIR / "model_comparison.json"
    comparison = json.loads(path.read_text()) if path.exists() else []
    return jsonify({
        "comparison": comparison,
        "registry_history": get_model_registry().history()[:10],
    })


@app.get("/explain")
def explain():
    """SHAP explanation. Defaults to the cached global summary; pass ?local=1
    for a per-prediction breakdown (slower for non-tree models)."""
    horizon = int(request.args.get("horizon", config.HORIZONS[0]))
    horizon_index = config.HORIZONS.index(horizon) if horizon in config.HORIZONS else 0

    if request.args.get("local", "").lower() not in ("1", "true", "yes"):
        summary = load_shap_summary()
        if summary is None:
            return jsonify({"error": "no SHAP summary yet -- run the training pipeline"}), 404
        return jsonify(summary)

    pipeline, metadata = inference._load_production()
    feat_cols = metadata.get("feature_columns") or []
    store_df = get_feature_store().read()
    if store_df.empty:
        return jsonify({"error": "feature store is empty"}), 404

    row = store_df.tail(1)
    background = store_df.tail(500)
    result = explain_prediction(
        pipeline, row, feat_cols, background=background,
        model_name=metadata.get("best_model", ""), horizon_index=horizon_index,
    )
    return jsonify(result)


@app.get("/alerts")
def alerts():
    result = inference.predict()
    active = result.get("alerts") or evaluate_forecast_alerts(result["forecast"])
    return jsonify({
        "threshold": config.ALERT_AQI_THRESHOLD,
        "active_count": len(active),
        "alerts": active,
    })


@app.get("/metrics")
def metrics():
    return Response(render_metrics(), mimetype="text/plain; version=0.0.4")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app.run(host="0.0.0.0", port=config.API_PORT, debug=False)


if __name__ == "__main__":
    main()
