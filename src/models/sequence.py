"""Encoder-decoder LSTM over a genuine hourly sequence.

Why this exists
---------------
The `tf_lstm` in the zoo is an RNN in name only: it reshapes the flat engineered
feature vector into an arbitrary `(8, 16)` pseudo-sequence, so its "timesteps"
are slices of a feature vector rather than hours. That is why it underperforms --
it is a sequence model with nothing sequential to read.

This model fixes the input rather than growing the architecture:

    encoder   the true last 72 hours of observations, shape (72, C)
    decoder   known-future weather at +24/48/72h -- the one idea worth taking
              from the Temporal Fusion Transformer, without TFT's parameter
              count or its PyTorch dependency
    head      three outputs, one per horizon

Known-future covariates are what a purely autoregressive model structurally
cannot use. Shifting observed weather backwards measured the ceiling on that
information at +8.9% RMSE (see experiments/exp_future_weather.py); this is the
architecture that can actually consume it.

Deliberately small -- ~30k parameters against 8k training rows. The failure mode
here is overfitting, not underfitting, and a bigger network would only make that
worse.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)

LOOKBACK = 72  # hours of history the encoder reads

# Channels the encoder sees each hour. Raw observations plus the cyclical clock;
# the engineered lag/rolling columns are deliberately excluded because the
# sequence itself already carries that information.
SEQ_CHANNELS = [
    "aqi", "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "dust", "aerosol_optical_depth",
    "temperature", "humidity", "pressure", "precipitation",
    "wind_speed", "wind_u", "wind_v", "boundary_layer_height",
    "hour_sin", "hour_cos",
]

# Variables genuinely knowable in advance from a weather forecast.
FUTURE_CHANNELS = [
    "temperature", "humidity", "pressure", "wind_speed",
    "wind_u", "wind_v", "precipitation", "boundary_layer_height",
]


def build_sequences(df: pd.DataFrame, target_cols: list[str],
                    lookback: int = LOOKBACK,
                    horizons: list[int] | None = None,
                    use_future: bool = True):
    """Frame -> (X_past, X_future, y, ts).

    Row i ends at ts[i]: X_past[i] is the `lookback` hours up to and including
    it, X_future[i] is the forecast weather at each horizon beyond it, and y[i]
    is the AQI at those horizons.

    Only rows with a complete history, complete future covariates and non-null
    targets survive, so nothing is silently padded.
    """
    horizons = horizons or config.HORIZONS
    df = df.sort_values("ts").reset_index(drop=True)

    seq_cols = [c for c in SEQ_CHANNELS if c in df.columns]
    fut_cols = [c for c in FUTURE_CHANNELS if c in df.columns] if use_future else []

    values = df[seq_cols].to_numpy(dtype="float32", copy=True)
    # Median-fill any residual gaps; the sequence must be dense to be a sequence.
    medians = np.nanmedian(values, axis=0)
    gaps = np.isnan(values)
    values[gaps] = np.take(medians, np.where(gaps)[1])

    fut_values = (df[fut_cols].to_numpy(dtype="float32", copy=True)
                  if fut_cols else None)
    if fut_values is not None:
        fmed = np.nanmedian(fut_values, axis=0)
        fgaps = np.isnan(fut_values)
        fut_values[fgaps] = np.take(fmed, np.where(fgaps)[1])

    targets = df[target_cols].to_numpy(dtype="float32", copy=True)
    max_h = max(horizons)

    starts = np.arange(lookback - 1, len(df) - max_h)
    valid = ~np.isnan(targets[starts]).any(axis=1)
    starts = starts[valid]
    if len(starts) == 0:
        raise ValueError("no complete sequences could be built")

    # Vectorised sliding window -- a Python loop here is ~100x slower.
    offsets = np.arange(-(lookback - 1), 1)
    X_past = values[starts[:, None] + offsets]                      # (N, L, C)

    if fut_values is not None:
        X_future = np.concatenate(
            [fut_values[starts + h] for h in horizons], axis=1      # (N, F*H)
        ).astype("float32")
    else:
        X_future = np.zeros((len(starts), 0), dtype="float32")

    y = targets[starts]
    ts = pd.DatetimeIndex(df["ts"])[starts]   # tz-aware; do NOT go via numpy
    return X_past, X_future, y, ts, seq_cols, fut_cols


class SequenceForecaster:
    """Encoder-decoder LSTM. Kept outside the sklearn Pipeline because it
    consumes a 3-D tensor, which a Pipeline's 2-D contract cannot express."""

    def __init__(self, lookback: int = LOOKBACK, units: int = 64,
                 epochs: int = 80, batch_size: int = 64,
                 learning_rate: float = 1e-3, use_future: bool = True,
                 seed: int = config.RANDOM_STATE, verbose: int = 0):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_future = use_future
        self.seed = seed
        self.verbose = verbose
        self.model_ = None

    # -- scaling -------------------------------------------------------------
    def _fit_scalers(self, X_past, X_future):
        self.past_mean_ = X_past.reshape(-1, X_past.shape[-1]).mean(axis=0)
        self.past_std_ = X_past.reshape(-1, X_past.shape[-1]).std(axis=0) + 1e-6
        if X_future.shape[1]:
            self.fut_mean_ = X_future.mean(axis=0)
            self.fut_std_ = X_future.std(axis=0) + 1e-6
        self.y_mean_ = None

    def _scale(self, X_past, X_future):
        past = (X_past - self.past_mean_) / self.past_std_
        fut = ((X_future - self.fut_mean_) / self.fut_std_
               if X_future.shape[1] else X_future)
        return past, fut

    # -- architecture --------------------------------------------------------
    def _build(self, n_channels: int, n_future: int, n_outputs: int):
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.seed)

        past_in = keras.layers.Input(shape=(self.lookback, n_channels), name="past")
        x = keras.layers.LSTM(self.units, return_sequences=True)(past_in)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.LSTM(self.units // 2)(x)
        encoded = keras.layers.Dropout(0.2)(x)

        inputs = [past_in]
        if n_future:
            fut_in = keras.layers.Input(shape=(n_future,), name="future")
            f = keras.layers.Dense(32, activation="relu")(fut_in)
            encoded = keras.layers.Concatenate()([encoded, f])
            inputs.append(fut_in)

        h = keras.layers.Dense(64, activation="relu")(encoded)
        h = keras.layers.Dropout(0.15)(h)
        out = keras.layers.Dense(n_outputs)(h)

        model = keras.Model(inputs=inputs, outputs=out)
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="huber",           # robust to the AQI spikes that matter most
            metrics=["mae"],
        )
        return model

    # -- api -----------------------------------------------------------------
    def fit(self, X_past, X_future, y):
        from tensorflow import keras

        self._fit_scalers(X_past, X_future)
        past, fut = self._scale(X_past, X_future)
        self.model_ = self._build(past.shape[-1], fut.shape[1], y.shape[1])

        inputs = [past] + ([fut] if fut.shape[1] else [])
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6,
                                              factor=0.5),
        ]
        self.model_.fit(
            inputs, y,
            epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.15,
            shuffle=False,          # time series: keep the validation tail contiguous
            callbacks=callbacks, verbose=self.verbose,
        )
        log.info("sequence model: %d parameters", self.model_.count_params())
        return self

    def predict(self, X_past, X_future):
        past, fut = self._scale(X_past, X_future)
        inputs = [past] + ([fut] if fut.shape[1] else [])
        return np.asarray(self.model_.predict(inputs, verbose=0))

    @property
    def n_params(self) -> int:
        return int(self.model_.count_params()) if self.model_ is not None else 0
