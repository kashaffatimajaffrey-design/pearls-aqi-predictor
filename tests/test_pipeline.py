"""Tests for the store abstractions and the model layer.

These use a tmp_path-scoped store so they never touch the real data directory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models import build_model, load_bundle, save_bundle
from src.pipelines.training_pipeline import _score, chronological_split
from src.store.feature_store import LocalFeatureStore
from src.store.model_registry import LocalModelRegistry


@pytest.fixture
def frame():
    n = 200
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "aqi": np.linspace(50, 150, n),
        "pm2_5": np.linspace(20, 90, n),
    })


class TestLocalFeatureStore:
    def test_write_then_read_round_trips(self, tmp_path, frame):
        store = LocalFeatureStore(tmp_path)
        assert store.write(frame) == len(frame)
        assert len(store.read()) == len(frame)

    def test_upsert_dedupes_on_timestamp(self, tmp_path, frame):
        store = LocalFeatureStore(tmp_path)
        store.write(frame)
        store.write(frame)                      # same window again
        assert len(store.read()) == len(frame)  # no duplicates

    def test_newer_rows_win_on_conflict(self, tmp_path, frame):
        store = LocalFeatureStore(tmp_path)
        store.write(frame)
        updated = frame.copy()
        updated["aqi"] = 999.0
        store.write(updated)
        assert (store.read()["aqi"] == 999.0).all()

    def test_appending_a_later_window_extends_the_store(self, tmp_path, frame):
        store = LocalFeatureStore(tmp_path)
        store.write(frame)
        later = frame.copy()
        later["ts"] = later["ts"] + pd.Timedelta(hours=len(frame))
        store.write(later)
        assert len(store.read()) == 2 * len(frame)

    def test_read_on_empty_store(self, tmp_path):
        assert LocalFeatureStore(tmp_path).read().empty

    def test_writing_nothing_is_a_noop(self, tmp_path):
        assert LocalFeatureStore(tmp_path).write(pd.DataFrame()) == 0


class TestLocalModelRegistry:
    def test_push_creates_a_version_and_production_copy(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "model.joblib").write_bytes(b"stub")

        registry = LocalModelRegistry(tmp_path / "registry")
        version = registry.push(bundle, {"best_model": "ridge", "metrics": {"rmse": 1.0}})

        assert version == "v0001"
        assert (registry.production_path() / "model.joblib").exists()

    def test_versions_increment_and_history_is_newest_first(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "model.joblib").write_bytes(b"stub")

        registry = LocalModelRegistry(tmp_path / "registry")
        registry.push(bundle, {"best_model": "ridge", "metrics": {"rmse": 2.0}})
        second = registry.push(bundle, {"best_model": "xgboost", "metrics": {"rmse": 1.0}})

        assert second == "v0002"
        history = registry.history()
        assert len(history) == 2
        assert history[0]["version"] == "v0002"

    def test_production_path_is_none_when_empty(self, tmp_path):
        assert LocalModelRegistry(tmp_path / "registry").production_path() is None


class TestModels:
    @staticmethod
    def _xy(n=300, n_features=6):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(
            rng.normal(size=(n, n_features)),
            columns=[f"f{i}" for i in range(n_features)],
        )
        # A learnable signal so a real model can beat a constant predictor.
        base = 2.5 * X["f0"] + 1.5 * X["f1"] - X["f2"]
        y = np.column_stack([base + rng.normal(0, .3, n) + k for k in (0, 1, 2)])
        return X, y

    @pytest.mark.parametrize("name", ["ridge", "random_forest", "gradient_boosting", "xgboost"])
    def test_fit_predict_shape(self, name):
        X, y = self._xy()
        model = build_model(name).fit(X, y)
        assert model.predict(X).shape == (len(X), len(config.HORIZONS))

    def test_ridge_learns_the_signal(self):
        X, y = self._xy()
        model = build_model("ridge").fit(X, y)
        pred = model.predict(X)
        # Correlated with truth well above chance.
        assert np.corrcoef(pred[:, 0], y[:, 0])[0, 1] > 0.9

    def test_persistence_baseline_repeats_the_current_value(self):
        X, y = self._xy()
        model = build_model("baseline_persistence")
        model.named_steps["model"].aqi_index = 0
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (len(X), 3)
        assert np.allclose(pred[:, 0], pred[:, 1])  # same value across horizons

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            build_model("does_not_exist")

    def test_bundle_round_trips(self, tmp_path):
        X, y = self._xy()
        model = build_model("ridge").fit(X, y)
        expected = model.predict(X)

        save_bundle(model, {"best_model": "ridge"}, tmp_path / "b")
        reloaded, metadata = load_bundle(tmp_path / "b")

        assert metadata["best_model"] == "ridge"
        assert np.allclose(reloaded.predict(X), expected)

    def test_handles_missing_values(self):
        """The imputer must absorb NaNs -- inference rows routinely have gaps."""
        X, y = self._xy()
        X.iloc[5:10, 0] = np.nan
        model = build_model("ridge").fit(X, y)
        assert np.isfinite(model.predict(X)).all()


class TestTrainingHelpers:
    def test_split_is_chronological(self, frame):
        train, test = chronological_split(frame, test_hours=24)
        assert len(test) == 24
        assert train["ts"].max() < test["ts"].min()

    def test_split_caps_test_size_on_small_frames(self, frame):
        train, test = chronological_split(frame, test_hours=10_000)
        assert len(train) > 0 and len(test) > 0

    def test_score_is_perfect_on_exact_predictions(self):
        y = np.random.default_rng(2).normal(size=(50, 3)) * 10 + 100
        scores = _score(y, y)
        assert scores["rmse"] == pytest.approx(0, abs=1e-9)
        assert scores["r2"] == pytest.approx(1.0)

    def test_score_reports_every_horizon(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=(50, 3))
        scores = _score(y, y + rng.normal(0, .1, (50, 3)))
        assert set(scores["per_horizon"]) == {f"{h}h" for h in config.HORIZONS}
        assert scores["rmse"] > 0


class TestColumnAlignment:
    """Regression tests for a real bug: an all-NaN column made SimpleImputer
    silently drop it, changing the matrix width so the persistence baseline read
    AQI from the wrong column and its RMSE went from 10.5 to 60.4."""

    @staticmethod
    def _xy_with_empty_column(n=200):
        rng = np.random.default_rng(7)
        X = pd.DataFrame({
            "empty_feature": np.full(n, np.nan),   # provider covers nothing here
            "aqi": rng.uniform(50, 150, n),
            "other": rng.normal(size=n),
        })
        y = np.column_stack([X["aqi"].to_numpy()] * 3)
        return X, y

    def test_imputer_preserves_column_count(self):
        from src.models import build_model

        X, y = self._xy_with_empty_column()
        model = build_model("ridge").fit(X, y)
        transformed = model.named_steps["impute"].transform(X)
        assert transformed.shape[1] == X.shape[1], "imputer must not drop columns"

    def test_persistence_baseline_survives_an_empty_column(self):
        from src.models import build_model

        X, y = self._xy_with_empty_column()
        model = build_model("baseline_persistence")
        model.named_steps["model"].aqi_index = list(X.columns).index("aqi")
        model.fit(X, y)
        # Persistence must reproduce the current AQI exactly.
        assert np.allclose(model.predict(X)[:, 0], X["aqi"].to_numpy())

    def test_feature_columns_excludes_all_nan(self):
        from src.features import feature_columns

        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
            "aqi": [1.0, 2, 3, 4, 5],
            "never_observed": [np.nan] * 5,
        })
        cols = feature_columns(df)
        assert "never_observed" not in cols
        assert "aqi" in cols
