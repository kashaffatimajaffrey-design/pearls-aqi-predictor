"""Tests for feature engineering.

The property that matters most here is causality: no feature may encode
information from the future. A leak would be invisible in the metrics (it makes
them look better) and fatal in production, so it is asserted directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import build_features, build_targets, feature_columns, latest_feature_row


@pytest.fixture
def raw():
    """Three weeks of hourly data with a realistic daily cycle and noise."""
    n = 24 * 21
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    daily = 20 * np.sin(2 * np.pi * np.arange(n) / 24)
    return pd.DataFrame({
        "ts": ts,
        "pm2_5": 45 + daily + rng.normal(0, 5, n),
        "pm10": 80 + daily + rng.normal(0, 8, n),
        "no2": 30 + rng.normal(0, 4, n),
        "so2": 12 + rng.normal(0, 2, n),
        "o3": 55 + rng.normal(0, 6, n),
        "co": 480 + rng.normal(0, 40, n),
        "temperature": 22 + 6 * np.sin(2 * np.pi * np.arange(n) / 24),
        "humidity": np.clip(60 + rng.normal(0, 10, n), 5, 100),
        "pressure": 1012 + rng.normal(0, 3, n),
        "wind_speed": np.abs(rng.normal(3.5, 1.5, n)),
        "wind_direction": rng.uniform(0, 360, n),
        "precipitation": np.zeros(n),
    })


class TestBuildFeatures:
    def test_produces_rows_and_core_columns(self, raw):
        feats = build_features(raw)
        assert not feats.empty
        for col in ("aqi", "hour_sin", "aqi_lag_24h", "aqi_change_rate_24h", "stagnation"):
            assert col in feats.columns

    def test_drops_the_lag_warmup_window(self, raw):
        feats = build_features(raw)
        # 72h of lags cannot be filled, so the output is shorter by at least that.
        assert len(feats) <= len(raw) - 72
        assert feats["aqi_lag_72h"].notna().all()

    def test_lag_features_are_actually_lagged(self, raw):
        feats = build_features(raw)
        # aqi_lag_24h at row i must equal aqi 24 rows earlier in the same frame.
        assert feats["aqi_lag_24h"].iloc[100] == pytest.approx(feats["aqi"].iloc[76])

    def test_no_lookahead_in_rolling_features(self, raw):
        """Rolling stats are shifted by one, so they cannot contain the current
        value -- this is the leak that would silently inflate every metric."""
        feats = build_features(raw)
        i = 200
        window = feats["aqi"].iloc[i - 24:i]
        assert feats["aqi_roll_mean_24h"].iloc[i] == pytest.approx(window.mean(), rel=1e-6)

    def test_cyclical_encoding_wraps(self, raw):
        feats = build_features(raw)
        assert feats["hour_sin"].between(-1, 1).all()
        assert (feats["hour_sin"] ** 2 + feats["hour_cos"] ** 2).round(6).eq(1).all()

    def test_wind_vector_matches_speed(self, raw):
        feats = build_features(raw)
        magnitude = np.sqrt(feats["wind_u"] ** 2 + feats["wind_v"] ** 2)
        assert magnitude.values == pytest.approx(feats["wind_speed"].values, rel=1e-5)

    def test_gaps_are_interpolated_onto_an_hourly_grid(self, raw):
        gapped = raw.drop(index=range(100, 104)).reset_index(drop=True)
        feats = build_features(gapped)
        deltas = feats["ts"].diff().dropna().unique()
        assert len(deltas) == 1 and deltas[0] == pd.Timedelta(hours=1)

    def test_empty_input_returns_empty(self):
        assert build_features(pd.DataFrame()).empty


class TestTargets:
    def test_targets_look_forward(self, raw):
        labelled = build_targets(build_features(raw))
        for horizon, col in zip(config.HORIZONS, config.TARGET_COLS, strict=False):
            assert col in labelled
            valid = labelled[col].notna()
            i = valid.idxmax()
            assert labelled[col].iloc[i] == pytest.approx(labelled["aqi"].iloc[i + horizon])

    def test_tail_targets_are_missing(self, raw):
        labelled = build_targets(build_features(raw))
        assert labelled[config.TARGET_COLS[-1]].tail(max(config.HORIZONS)).isna().all()


class TestFeatureColumns:
    def test_excludes_targets_and_identifiers(self, raw):
        labelled = build_targets(build_features(raw))
        cols = feature_columns(labelled)
        assert not set(cols) & set(config.TARGET_COLS)
        for banned in ("ts", "city", "dominant_pollutant", "feature_version"):
            assert banned not in cols

    def test_all_selected_columns_are_numeric(self, raw):
        labelled = build_targets(build_features(raw))
        for col in feature_columns(labelled):
            assert pd.api.types.is_numeric_dtype(labelled[col])


class TestLatestFeatureRow:
    def test_returns_one_complete_row(self, raw):
        row = latest_feature_row(raw)
        assert len(row) == 1
        assert row["aqi"].notna().all()

    def test_uses_the_newest_timestamp(self, raw):
        row = latest_feature_row(raw)
        assert row["ts"].iloc[0] == raw["ts"].max()
