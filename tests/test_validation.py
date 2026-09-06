"""Tests for raw-input validation.

These guard the boundary where bad upstream data would otherwise enter the
training set silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validation import BOUNDS, STUCK_SENSOR_HOURS, validate


def frame(n=48, **overrides):
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "pm2_5": np.linspace(20, 60, n),
        "pm10": np.linspace(40, 100, n),
        "temperature": np.linspace(18, 32, n),
        "humidity": np.linspace(40, 80, n),
        "wind_speed": np.linspace(1, 6, n),
        "pressure": np.full(n, 1012.0),
    })
    for col, values in overrides.items():
        df[col] = values
    return df


class TestBounds:
    def test_clean_data_passes(self):
        out, report = validate(frame())
        assert report["total_rejected"] == 0
        assert out["pm2_5"].notna().all()

    def test_negative_concentration_is_nulled(self):
        df = frame()
        df.loc[5, "pm2_5"] = -10
        out, report = validate(df)
        assert np.isnan(out.loc[5, "pm2_5"])
        assert report["rejected"]["pm2_5"] == 1

    def test_absurd_spike_is_nulled(self):
        """A decimal-point error must not become a training target."""
        df = frame()
        df.loc[7, "pm2_5"] = 9_000
        out, report = validate(df)
        assert np.isnan(out.loc[7, "pm2_5"])

    def test_impossible_humidity_is_nulled(self):
        df = frame()
        df.loc[3, "humidity"] = 250
        out, _ = validate(df)
        assert np.isnan(out.loc[3, "humidity"])

    def test_only_the_bad_column_is_nulled(self):
        """A bad NO2 reading must not cost us a good hour of PM2.5."""
        df = frame()
        df["no2"] = 30.0
        df.loc[4, "no2"] = -5
        out, _ = validate(df)
        assert np.isnan(out.loc[4, "no2"])
        assert not np.isnan(out.loc[4, "pm2_5"])

    def test_row_count_is_preserved(self):
        df = frame()
        df.loc[2, "pm2_5"] = -1
        out, _ = validate(df)
        assert len(out) == len(df)

    @pytest.mark.parametrize("col", list(BOUNDS))
    def test_every_bound_is_ordered(self, col):
        lo, hi = BOUNDS[col]
        assert lo < hi


class TestStuckSensor:
    def test_flags_a_frozen_sensor(self):
        df = frame(n=60)
        df.loc[:, "pm2_5"] = 42.0     # never changes
        _, report = validate(df)
        assert any("pm2_5" in w for w in report["warnings"])

    def test_normal_variation_is_not_flagged(self):
        _, report = validate(frame(n=60))
        assert not any("pm2_5 repeated" in w for w in report["warnings"])

    def test_short_constant_run_is_tolerated(self):
        df = frame(n=60)
        df.loc[0:STUCK_SENSOR_HOURS - 5, "pm2_5"] = 30.0
        _, report = validate(df)
        assert not any("pm2_5 repeated" in w for w in report["warnings"])


class TestTimestamps:
    def test_duplicates_are_dropped(self):
        df = frame(n=10)
        df.loc[5, "ts"] = df.loc[4, "ts"]
        out, report = validate(df)
        assert len(out) == 9
        assert any("duplicate" in w for w in report["warnings"])

    def test_future_rows_are_dropped(self):
        df = frame(n=10)
        df.loc[9, "ts"] = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=5)
        out, report = validate(df)
        assert len(out) == 9
        assert any("future" in w for w in report["warnings"])


class TestCoverage:
    def test_mostly_missing_is_warned(self):
        df = frame(n=20)
        df.loc[0:14, "pm2_5"] = np.nan
        _, report = validate(df)
        assert any("missing" in w for w in report["warnings"])

    def test_strict_mode_raises_on_bad_coverage(self):
        df = frame(n=20)
        df.loc[0:14, "pm2_5"] = np.nan
        with pytest.raises(ValueError):
            validate(df, strict=True)

    def test_empty_frame_is_handled(self):
        out, report = validate(pd.DataFrame())
        assert report["status"] == "empty"
        assert out.empty
