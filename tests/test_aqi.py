"""Tests for the EPA AQI computation.

The breakpoint table is the one piece of this project where a silent error would
be invisible downstream -- every model target derives from it -- so the anchor
points are pinned against the published EPA values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aqi import (
    advice,
    categorize,
    category_color,
    compute_aqi_frame,
    sub_index,
    ugm3_to_ppb,
)


class TestSubIndex:
    @pytest.mark.parametrize(
        "concentration, expected",
        [
            (0.0, 0),      # bottom of the scale
            (9.0, 50),     # top of "Good" under the 2024 revision
            (35.4, 100),   # top of "Moderate"
            (55.4, 150),
            (125.4, 200),
        ],
    )
    def test_pm25_breakpoints(self, concentration, expected):
        assert sub_index(concentration, "pm2_5") == pytest.approx(expected, abs=1)

    def test_pm25_is_monotonic(self):
        values = [sub_index(c, "pm2_5") for c in range(0, 300, 5)]
        assert values == sorted(values)

    def test_pm10_breakpoints(self):
        assert sub_index(54, "pm10") == pytest.approx(50, abs=1)
        assert sub_index(154, "pm10") == pytest.approx(100, abs=1)

    def test_above_table_caps_at_500(self):
        assert sub_index(10_000, "pm2_5") == 500

    def test_missing_input_returns_nan(self):
        assert np.isnan(sub_index(float("nan"), "pm2_5"))
        assert np.isnan(sub_index(None, "pm2_5"))

    def test_negative_treated_as_zero(self):
        assert sub_index(-5, "pm2_5") == 0


class TestUnitConversion:
    def test_o3_ugm3_to_ppb(self):
        # 100 ug/m3 O3 -> 100 * 24.45 / 48 ~ 50.9 ppb
        assert ugm3_to_ppb(100, "o3") == pytest.approx(50.9, abs=0.5)

    def test_co_is_returned_in_ppm(self):
        # CO uses ppm, so the result must be ~1000x smaller than the ppb value.
        assert ugm3_to_ppb(1000, "co") == pytest.approx(0.873, abs=0.02)


class TestCategories:
    @pytest.mark.parametrize(
        "aqi, expected",
        [
            (25, "Good"),
            (75, "Moderate"),
            (125, "Unhealthy for Sensitive Groups"),
            (175, "Unhealthy"),
            (250, "Very Unhealthy"),
            (400, "Hazardous"),
        ],
    )
    def test_categorize(self, aqi, expected):
        assert categorize(aqi) == expected

    def test_boundaries_are_inclusive_at_the_top(self):
        assert categorize(50) == "Good"
        assert categorize(51) == "Moderate"

    def test_every_category_has_a_colour_and_advice(self):
        for aqi in (25, 75, 125, 175, 250, 400):
            assert category_color(aqi).startswith("#")
            assert len(advice(aqi)) > 10

    def test_nan_is_unknown(self):
        assert categorize(float("nan")) == "Unknown"


class TestComputeAqiFrame:
    @staticmethod
    def _frame(hours=48, pm25=35.0):
        return pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC"),
            "pm2_5": np.full(hours, pm25),
            "pm10": np.full(hours, 60.0),
            "no2": np.full(hours, 30.0),
            "so2": np.full(hours, 10.0),
            "o3": np.full(hours, 60.0),
            "co": np.full(hours, 500.0),
        })

    def test_adds_aqi_and_dominant_pollutant(self):
        out = compute_aqi_frame(self._frame())
        assert "aqi" in out and "dominant_pollutant" in out
        assert out["aqi"].notna().all()

    def test_aqi_is_the_max_of_sub_indices(self):
        out = compute_aqi_frame(self._frame())
        sub_cols = [c for c in out.columns if c.startswith("aqi_")]
        assert (out["aqi"] == out[sub_cols].max(axis=1)).all()

    def test_dominant_pollutant_names_the_max(self):
        # Extreme PM2.5 with everything else low must make PM2.5 dominant.
        df = self._frame(pm25=250.0)
        df["pm10"] = 20.0
        out = compute_aqi_frame(df)
        assert out["dominant_pollutant"].iloc[-1] == "pm2_5"

    def test_higher_pollution_gives_higher_aqi(self):
        low = compute_aqi_frame(self._frame(pm25=8.0))["aqi"].iloc[-1]
        high = compute_aqi_frame(self._frame(pm25=150.0))["aqi"].iloc[-1]
        assert high > low

    def test_raises_without_pollutants(self):
        with pytest.raises(ValueError):
            compute_aqi_frame(pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"], utc=True)}))
