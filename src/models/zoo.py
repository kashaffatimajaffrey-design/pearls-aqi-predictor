"""Candidate models, all wrapped to a single fit/predict contract.

Every estimator maps a feature row to a 3-vector -- AQI at +24h, +48h and +72h.
The families deliberately span the range the brief asks for:

    baseline_persistence  statistical, no learning (the bar to beat)
    ridge                 linear, L2-regularised
    random_forest         bagged trees
    gradient_boosting     sklearn boosting
    xgboost               gradient boosting, the usual tabular winner
    tf_mlp                TensorFlow dense net
    tf_lstm               TensorFlow sequence model over the lag window

Scaling is folded into a Pipeline where the estimator needs it, so a bundle is
always self-contained: load it, call predict, done.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config

log = logging.getLogger(__name__)


# ------------------------------------------------------------- baselines ----
class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Tomorrow looks like today. A forecasting model that cannot beat this is
    not learning anything, so it is trained and scored alongside the rest."""

    def __init__(self, aqi_index: int = 0):
        self.aqi_index = aqi_index

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_outputs_ = np.asarray(y).shape[1] if y is not None and np.ndim(y) > 1 else len(config.HORIZONS)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        current = X[:, self.aqi_index]
        return np.repeat(current[:, None], self.n_outputs_, axis=1)


# --------------------------------------------------------------- keras -----
class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    """Minimal sklearn-style wrapper around a Keras model.

    Kept in-house rather than pulling scikeras: it is ~40 lines, and it lets the
    bundle serialise the network as `.keras` next to the joblib preprocessing.
    """

    def __init__(self, architecture: str = "mlp", n_lags: int = 8, epochs: int = 60,
                 batch_size: int = 64, learning_rate: float = 1e-3, verbose: int = 0):
        self.architecture = architecture
        self.n_lags = n_lags
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.model_ = None

    # -- construction --------------------------------------------------------
    def _reshape(self, X: np.ndarray) -> np.ndarray:
        """For the LSTM, present the feature vector as a short pseudo-sequence.

        Zero-padding happens here in NumPy rather than in a Lambda layer, so the
        saved `.keras` file contains no serialised Python closure.
        """
        if self.architecture != "lstm":
            return X
        pad = self.steps_ * self.per_step_ - X.shape[1]
        if pad:
            X = np.pad(X, ((0, 0), (0, pad)))
        return X.reshape(-1, self.steps_, self.per_step_)

    def _build(self, n_features: int, n_outputs: int):
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(config.RANDOM_STATE)
        if self.architecture == "lstm":
            model = keras.Sequential([
                keras.layers.Input(shape=(self.steps_, self.per_step_)),
                keras.layers.LSTM(96, return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(48),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(n_outputs),
            ])
        else:
            model = keras.Sequential([
                keras.layers.Input(shape=(n_features,)),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(n_outputs),
            ])

        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="huber",           # robust to the AQI spikes that matter most
            metrics=["mae"],
        )
        return model

    # -- sklearn API ---------------------------------------------------------
    def fit(self, X, y):
        from tensorflow import keras

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.n_features_in_ = X.shape[1]
        self.steps_ = self.n_lags
        self.per_step_ = int(np.ceil(X.shape[1] / self.steps_))
        self.model_ = self._build(X.shape[1], y.shape[1])
        X = self._reshape(X)
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5),
        ]
        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            shuffle=False,          # time series: keep the validation tail contiguous
            callbacks=callbacks,
            verbose=self.verbose,
        )
        return self

    def predict(self, X):
        X = self._reshape(np.asarray(X, dtype="float32"))
        return np.asarray(self.model_.predict(X, verbose=0))

    # -- persistence ---------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["model_"] = None      # the network is saved separately as .keras
        return state


# ------------------------------------------------------------- builders ----
def _wrap(estimator, *, scale: bool, multioutput: bool) -> Pipeline:
    """Impute -> (optionally) scale -> estimator, as one picklable object."""
    # keep_empty_features is load-bearing: without it SimpleImputer silently
    # DROPS an all-NaN column, changing the matrix width and invalidating any
    # positional feature index downstream (this cost us a broken baseline once).
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    if multioutput:
        estimator = MultiOutputRegressor(estimator)
    steps.append(("model", estimator))
    return Pipeline(steps)


MODEL_BUILDERS = {
    "baseline_persistence": lambda: _wrap(
        PersistenceBaseline(), scale=False, multioutput=False
    ),
    "ridge": lambda: _wrap(
        Ridge(alpha=5.0, random_state=config.RANDOM_STATE),
        scale=True, multioutput=False,
    ),
    "random_forest": lambda: _wrap(
        RandomForestRegressor(
            n_estimators=350, max_depth=18, min_samples_leaf=3,
            n_jobs=-1, random_state=config.RANDOM_STATE,
        ),
        scale=False, multioutput=False,
    ),
    "gradient_boosting": lambda: _wrap(
        GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=4,
            subsample=0.9, random_state=config.RANDOM_STATE,
        ),
        scale=False, multioutput=True,
    ),
    "xgboost": lambda: _wrap(_xgb(), scale=False, multioutput=False),
    "tf_mlp": lambda: _wrap(
        KerasRegressorWrapper(architecture="mlp"), scale=True, multioutput=False
    ),
    "tf_lstm": lambda: _wrap(
        KerasRegressorWrapper(architecture="lstm"), scale=True, multioutput=False
    ),
}


def _xgb():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=600, learning_rate=0.04, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.5, min_child_weight=3,
        objective="reg:squarederror", tree_method="hist",
        n_jobs=-1, random_state=config.RANDOM_STATE,
    )


def build_model(name: str):
    if name not in MODEL_BUILDERS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name]()


def _keras_step(pipeline) -> KerasRegressorWrapper | None:
    step = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
    return step if isinstance(step, KerasRegressorWrapper) else None


# ---------------------------------------------------------------- bundle ----
def save_bundle(pipeline, metadata: dict, out_dir: Path) -> Path:
    """Persist estimator + metadata to a self-contained directory."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keras_step = _keras_step(pipeline)
    if keras_step is not None and keras_step.model_ is not None:
        keras_step.model_.save(out_dir / "keras_model.keras")

    joblib.dump(pipeline, out_dir / "model.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return out_dir


def load_bundle(bundle_dir: Path):
    """Load a bundle back into (pipeline, metadata)."""
    bundle_dir = Path(bundle_dir)
    pipeline = joblib.load(bundle_dir / "model.joblib")

    keras_path = bundle_dir / "keras_model.keras"
    keras_step = _keras_step(pipeline)
    if keras_step is not None and keras_path.exists():
        from tensorflow import keras

        keras_step.model_ = keras.models.load_model(keras_path, compile=False)

    meta_path = bundle_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pipeline, metadata
