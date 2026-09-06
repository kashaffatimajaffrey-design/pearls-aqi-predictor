"""Historical backfill.

Replays the feature pipeline over past dates to create the training set. The
window is walked in chunks with an overlap of `OVERLAP_HOURS` so that every
chunk has enough prior context to fill its 72-hour lags; overlapping rows
de-duplicate on write.

    python -m src.pipelines.backfill --days 365
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta

import pandas as pd

from .. import config
from ..data import fetch_range
from ..features import build_features
from ..store import get_feature_store

log = logging.getLogger(__name__)

CHUNK_DAYS = 30
OVERLAP_HOURS = 96  # > the longest lag (72h) so each chunk can warm up


def run(days: int = config.BACKFILL_DAYS, source: str | None = None,
        chunk_days: int = CHUNK_DAYS) -> dict:
    store = get_feature_store()
    end = datetime.now(UTC)
    start = end - timedelta(days=days)

    all_raw: list[pd.DataFrame] = []
    cursor = start
    chunk_no = 0
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        fetch_start = cursor - timedelta(hours=OVERLAP_HOURS)
        chunk_no += 1
        log.info("chunk %d: %s -> %s", chunk_no, fetch_start.date(), chunk_end.date())
        try:
            raw = fetch_range(fetch_start, chunk_end, source=source)
            if not raw.empty:
                all_raw.append(raw)
                log.info("  %d rows", len(raw))
            else:
                log.warning("  empty chunk")
        except Exception as exc:
            log.warning("  chunk failed (%s) -- continuing", exc)
        cursor = chunk_end

    if not all_raw:
        raise RuntimeError("backfill returned no data at all")

    raw = (
        pd.concat(all_raw, ignore_index=True)
        .drop_duplicates(subset="ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    raw.to_parquet(config.RAW_DIR / "backfill_raw.parquet", index=False)

    features = build_features(raw)
    if features.empty:
        raise RuntimeError("backfill produced no usable feature rows")

    written = store.write(features)

    summary = {
        "backfilled_at": datetime.now(UTC).isoformat(),
        "requested_days": days,
        "source_configured": source or config.resolved_source(),
        "source_effective": raw.attrs.get("source", "unknown"),
        "feature_store": store.name,
        "raw_rows": int(len(raw)),
        "feature_rows": int(written),
        "coverage_start": str(features["ts"].min()),
        "coverage_end": str(features["ts"].max()),
        "coverage_pct": round(
            100 * len(features) / max(1, int((end - start).total_seconds() // 3600)), 1
        ),
    }
    (config.ARTIFACT_DIR / "last_backfill.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    log.info("backfill complete: %s", summary)
    return summary


def main() -> dict:
    parser = argparse.ArgumentParser(description="Historical AQI feature backfill")
    parser.add_argument("--days", type=int, default=config.BACKFILL_DAYS)
    parser.add_argument("--source", default=None, choices=["openweather", "openmeteo"])
    parser.add_argument("--chunk-days", type=int, default=CHUNK_DAYS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run(days=args.days, source=args.source, chunk_days=args.chunk_days)
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
