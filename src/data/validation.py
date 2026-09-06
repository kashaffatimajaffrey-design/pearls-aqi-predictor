"""Input validation for raw observations.

Upstream APIs occasionally return physically impossible values -- negative
concentrations, a stuck sensor repeating one number for days, a decimal-point
error putting PM2.5 at 9000. Without a gate, those flow through feature
engineering into the training set and quietly corrupt the model; the pipeline
reports success the whole way.

The policy here is deliberately conservative: implausible values are nulled
(not the whole row dropped), because a bad NO2 reading should not cost us an
otherwise good hour of PM2.5. Downstream interpolation and the median imputer
then handle the holes. Anything rejected is counted and reported so the problem
is visible rather than silent.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# (min, max) plausible values. Bounds are generous -- the aim is to catch broken
# data, not to clip genuine pollution episodes. The PM2.5 ceiling sits above the
# worst readings ever recorded in Delhi or Lahore.
BOUNDS: dict[str, tuple[float, float]] = {
    "pm2_5": (0.0, 1000.0),      # ug/m3
    "pm10": (0.0, 2000.0),
    "no2": (0.0, 1000.0),
    "so2": (0.0, 1000.0),
    "o3": (0.0, 800.0),
    "co": (0.0, 50000.0),
    "dust": (0.0, 5000.0),                  # ug/m3; dust storms reach the low thousands
    "ammonia": (0.0, 500.0),
    "aerosol_optical_depth": (0.0, 10.0),   # dimensionless
    "methane": (0.0, 10000.0),              # ug/m3
    "temperature": (-60.0, 60.0),   # degrees C
    "humidity": (0.0, 100.0),       # percent
    "pressure": (800.0, 1100.0),    # hPa
    "wind_speed": (0.0, 150.0),     # m/s
    "wind_direction": (0.0, 360.0), # degrees
    "precipitation": (0.0, 500.0),  # mm/h
    "boundary_layer_height": (0.0, 6000.0), # metres
}

# A sensor repeating the identical value this many hours is almost certainly
# stuck rather than measuring genuinely constant air.
STUCK_SENSOR_HOURS = 24


def validate(df: pd.DataFrame, *, strict: bool = False) -> tuple[pd.DataFrame, dict]:
    """Null out implausible values and report what was rejected.

    Returns (cleaned_frame, report). With `strict=True`, raises when the data is
    bad enough that training on it would be a mistake.
    """
    if df is None or df.empty:
        return df, {"rows": 0, "status": "empty"}

    out = df.copy()
    report: dict = {"rows": int(len(out)), "rejected": {}, "warnings": []}

    # -- 1. physically impossible values ------------------------------------
    for col, (lo, hi) in BOUNDS.items():
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        bad = values.notna() & ((values < lo) | (values > hi))
        n_bad = int(bad.sum())
        if n_bad:
            out.loc[bad, col] = np.nan
            report["rejected"][col] = n_bad
            log.warning("%d out-of-range %s value(s) nulled (bounds %s-%s)",
                        n_bad, col, lo, hi)
        else:
            out[col] = values

    # -- 2. stuck sensors ----------------------------------------------------
    for col in ("pm2_5", "pm10", "temperature"):
        if col not in out.columns or out[col].notna().sum() < STUCK_SENSOR_HOURS:
            continue
        # Longest run of identical consecutive values.
        series = out[col].dropna()
        run_ids = (series != series.shift()).cumsum()
        longest = int(run_ids.value_counts().max()) if len(run_ids) else 0
        if longest >= STUCK_SENSOR_HOURS:
            msg = f"{col} repeated the same value for {longest} consecutive hours"
            report["warnings"].append(msg)
            log.warning("possible stuck sensor: %s", msg)

    # -- 3. coverage ---------------------------------------------------------
    if "pm2_5" in out.columns:
        missing_pct = float(out["pm2_5"].isna().mean() * 100)
        report["pm2_5_missing_pct"] = round(missing_pct, 2)
        if missing_pct > 50:
            msg = f"PM2.5 is {missing_pct:.0f}% missing"
            report["warnings"].append(msg)
            if strict:
                raise ValueError(f"input validation failed: {msg}")

    # -- 4. timestamp sanity -------------------------------------------------
    if "ts" in out.columns:
        ts = pd.to_datetime(out["ts"], utc=True)
        if ts.duplicated().any():
            n = int(ts.duplicated().sum())
            report["warnings"].append(f"{n} duplicate timestamp(s) dropped")
            out = out.loc[~ts.duplicated()].reset_index(drop=True)
        future = ts > pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)
        if future.any():
            report["warnings"].append(f"{int(future.sum())} future timestamp(s) dropped")
            out = out.loc[~future.to_numpy()].reset_index(drop=True)

    total_rejected = sum(report["rejected"].values())
    report["total_rejected"] = total_rejected
    report["status"] = "ok" if not report["warnings"] and not total_rejected else "warnings"

    if total_rejected:
        log.info("validation nulled %d value(s) across %d row(s)", total_rejected, len(out))
    return out, report
