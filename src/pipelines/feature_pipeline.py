"""Hourly feature pipeline.

Fetch -> engineer -> write to the Feature Store -> check alert thresholds.

Runs every hour under GitHub Actions. It pulls a 7-day trailing window rather
than a single hour so that all lag/rolling features can be rebuilt from scratch;
the Feature Store upsert on `ts` means re-processed hours simply refresh, which
also self-heals any gap left by a failed earlier run.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import pandas as pd

from .. import config
from ..alerts import evaluate_observation_alert
from ..data import fetch_recent
from ..features import build_features
from ..monitoring.metrics import record_pipeline_run
from ..store import get_feature_store

log = logging.getLogger(__name__)


def run(hours: int = 168, source: str | None = None) -> dict:
    started = datetime.now(UTC)
    store = get_feature_store()

    raw = fetch_recent(hours=hours, source=source)
    if raw.empty:
        raise RuntimeError("no raw observations returned by any source")

    raw_path = config.RAW_DIR / "latest_raw.parquet"
    raw.to_parquet(raw_path, index=False)

    features = build_features(raw)
    if features.empty:
        raise RuntimeError(
            f"feature engineering produced no rows from {len(raw)} observations "
            "(window too short to fill the 72h lags?)"
        )

    written = store.write(features)

    latest = features.iloc[-1]
    alert = evaluate_observation_alert(float(latest["aqi"]), pd.Timestamp(latest["ts"]))

    summary = {
        "run_at": started.isoformat(),
        "duration_s": round((datetime.now(UTC) - started).total_seconds(), 2),
        "source_configured": source or config.resolved_source(),
        "source_effective": raw.attrs.get("source", "unknown"),
        "feature_store": store.name,
        "raw_rows": int(len(raw)),
        "feature_rows_written": int(written),
        "feature_count": int(len(features.columns)),
        "window_start": str(features["ts"].min()),
        "window_end": str(features["ts"].max()),
        "latest_aqi": round(float(latest["aqi"]), 1),
        "dominant_pollutant": str(latest.get("dominant_pollutant", "n/a")),
        "alert": alert,
    }

    (config.ARTIFACT_DIR / "last_feature_run.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    record_pipeline_run("feature", summary["duration_s"], ok=True)
    log.info("feature pipeline complete: %s", summary)
    return summary


def main() -> dict:
    parser = argparse.ArgumentParser(description="Hourly AQI feature pipeline")
    parser.add_argument("--hours", type=int, default=168,
                        help="trailing window to fetch (default 168 = 7 days)")
    parser.add_argument("--source", default=None, choices=["openweather", "openmeteo"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run(hours=args.hours, source=args.source)
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
