"""Daily training pipeline.

Read features -> build targets -> train every candidate -> score on a held-out
time tail -> register the winner -> compute SHAP.

Two things worth calling out:

* The split is chronological, never random. A shuffled split on a time series
  leaks the future into training through the lag features and produces R2
  numbers that do not survive contact with reality.
* Model selection uses mean RMSE across the three horizons. The persistence
  baseline is scored too, and a candidate is only promoted if it beats it --
  otherwise we would happily ship a model that has learned nothing.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .. import config
from ..explain import compute_shap
from ..features import add_future_weather, build_targets, feature_columns
from ..models import build_model, save_bundle
from ..monitoring.metrics import record_model_metrics, record_pipeline_run
from ..store import get_feature_store, get_model_registry

log = logging.getLogger(__name__)


class ModelRejected(RuntimeError):
    """Raised when a freshly trained model fails the promotion gate."""

DEFAULT_MODELS = [
    "baseline_persistence",
    "ridge",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "tf_mlp",
    "tf_lstm",
]

# Trained through a separate path: these consume a 3-D (samples, hours,
# channels) tensor, which the sklearn Pipeline 2-D contract cannot carry.
# Scored on the identical hold-out window so the comparison stays fair.
SEQUENCE_MODELS = ["tf_seq_lstm", "tf_seq_lstm_nofuture"]


# ------------------------------------------------------------------ utils ---
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE / MAE / R2, overall and per horizon."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    per_horizon = {}
    for i, horizon in enumerate(config.HORIZONS):
        yt, yp = y_true[:, i], y_pred[:, i]
        per_horizon[f"{horizon}h"] = {
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
        }

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "per_horizon": per_horizon,
    }


def load_training_frame() -> tuple[pd.DataFrame, list[str]]:
    store = get_feature_store()
    features = store.read()
    if features.empty:
        raise RuntimeError(
            "feature store is empty -- run `python -m src.pipelines.backfill` first"
        )

    # Known-future weather. At training this is observed weather shifted back
    # (perfect prog) -- an optimistic upper bound. Inference substitutes a real
    # forecast, so live accuracy will sit below the hold-out number; that gap is
    # recorded in the model metadata rather than left implicit.
    features = add_future_weather(features)
    labelled = build_targets(features).dropna(subset=config.TARGET_COLS)
    if len(labelled) < 200:
        raise RuntimeError(
            f"only {len(labelled)} labelled rows available; need at least 200. "
            "Backfill a longer window."
        )
    return labelled, feature_columns(labelled)


def chronological_split(df: pd.DataFrame, test_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("ts").reset_index(drop=True)
    test_hours = min(test_hours, max(24, len(df) // 5))
    cut = len(df) - test_hours
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()



def _train_sequence_models(labelled, names, split_ts, results, fitted):
    """Train the encoder-decoder LSTMs and score them on the same test window.

    The split is applied by timestamp rather than row position, because building
    sequences drops the first `lookback` rows and shifts every index.
    """
    from ..models.sequence import SequenceForecaster, build_sequences

    for name in names:
        use_future = not name.endswith("_nofuture")
        log.info("--- training %s (use_future=%s) ---", name, use_future)
        t0 = datetime.now(UTC)
        try:
            X_past, X_future, y, ts, seq_cols, fut_cols = build_sequences(
                labelled, config.TARGET_COLS, use_future=use_future
            )
            mask = np.asarray(ts < pd.Timestamp(split_ts))
            if mask.sum() < 200 or (~mask).sum() < 24:
                raise ValueError("not enough sequences either side of the split")

            model = SequenceForecaster(use_future=use_future)
            model.fit(X_past[mask], X_future[mask], y[mask])

            test_metrics = _score(y[~mask], model.predict(X_past[~mask], X_future[~mask]))
            train_metrics = _score(y[mask], model.predict(X_past[mask], X_future[mask]))

            results[name] = {
                "status": "ok",
                "train": train_metrics,
                "test": test_metrics,
                "fit_seconds": round((datetime.now(UTC) - t0).total_seconds(), 1),
                "architecture": "encoder-decoder LSTM",
                "n_params": model.n_params,
                "sequence_channels": len(seq_cols),
                "future_channels": len(fut_cols),
                "n_test_sequences": int((~mask).sum()),
            }
            fitted[name] = model
            log.info("%-22s test RMSE %6.2f  MAE %6.2f  R2 %6.3f  (%d params)",
                     name, test_metrics["rmse"], test_metrics["mae"],
                     test_metrics["r2"], model.n_params)
        except Exception as exc:
            log.warning("%s failed: %s", name, exc)
            results[name] = {"status": "failed", "error": str(exc)}



def _sequence_verdict(trained: dict, seq_winner: str | None, best_metrics: dict) -> dict | None:
    """Report how the sequence models fared against the promoted tabular model.

    They are compared but not promotable (save_bundle assumes an sklearn
    Pipeline), so if one wins we say so explicitly rather than quietly shipping
    the runner-up.
    """
    if seq_winner is None:
        return None
    seq = trained[seq_winner]["test"]
    beat = seq["rmse"] < best_metrics["rmse"]
    if beat:
        log.warning(
            "%s (RMSE %.2f) BEAT the promoted tabular model (RMSE %.2f) but is "
            "not yet promotable -- bundle serialisation needs a sequence path",
            seq_winner, seq["rmse"], best_metrics["rmse"],
        )
    return {
        "best_sequence_model": seq_winner,
        "rmse": seq["rmse"],
        "r2": seq["r2"],
        "beat_promoted_model": bool(beat),
        "promotable": False,
        "n_params": trained[seq_winner].get("n_params"),
    }


# ------------------------------------------------------------------- run -----
def run(models: list[str] | None = None, test_hours: int | None = None,
        skip_shap: bool = False, force: bool = False) -> dict:
    started = datetime.now(UTC)
    models = models or DEFAULT_MODELS
    test_hours = test_hours or config.TEST_SIZE_HOURS

    labelled, feat_cols = load_training_frame()
    train_df, test_df = chronological_split(labelled, test_hours)

    X_train = train_df[feat_cols]
    y_train = train_df[config.TARGET_COLS].to_numpy(dtype=float)
    X_test = test_df[feat_cols]
    y_test = test_df[config.TARGET_COLS].to_numpy(dtype=float)

    log.info(
        "training on %d rows x %d features; holding out %d rows (%s -> %s)",
        len(X_train), len(feat_cols), len(X_test),
        test_df["ts"].min(), test_df["ts"].max(),
    )

    # PersistenceBaseline reads current AQI positionally, so tell it where it is.
    aqi_index = feat_cols.index("aqi") if "aqi" in feat_cols else 0

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    for name in [m for m in models if m not in SEQUENCE_MODELS]:
        log.info("--- training %s ---", name)
        t0 = datetime.now(UTC)
        try:
            pipeline = build_model(name)
            if name == "baseline_persistence":
                pipeline.named_steps["model"].aqi_index = aqi_index
            pipeline.fit(X_train, y_train)

            train_metrics = _score(y_train, pipeline.predict(X_train))
            test_metrics = _score(y_test, pipeline.predict(X_test))

            results[name] = {
                "status": "ok",
                "train": train_metrics,
                "test": test_metrics,
                "fit_seconds": round((datetime.now(UTC) - t0).total_seconds(), 1),
            }
            fitted[name] = pipeline
            log.info(
                "%-22s test RMSE %6.2f  MAE %6.2f  R2 %6.3f",
                name, test_metrics["rmse"], test_metrics["mae"], test_metrics["r2"],
            )
        except Exception as exc:
            log.warning("%s failed: %s", name, exc)
            results[name] = {"status": "failed", "error": str(exc)}

    seq_names = [m for m in models if m in SEQUENCE_MODELS]
    if seq_names:
        _train_sequence_models(labelled, seq_names, test_df["ts"].min(), results, fitted)

    trained = {k: v for k, v in results.items() if v.get("status") == "ok"}
    if not trained:
        raise RuntimeError("every candidate model failed to train")

    baseline_rmse = trained.get("baseline_persistence", {}).get("test", {}).get("rmse")
    # Sequence models are compared but not yet promotable: save_bundle assumes an
    # sklearn Pipeline. If one wins, that is reported loudly rather than shipped.
    promotable = {
        k: v for k, v in trained.items()
        if k != "baseline_persistence" and k not in SEQUENCE_MODELS
    }
    seq_winner = min(
        (k for k in trained if k in SEQUENCE_MODELS),
        key=lambda k: trained[k]["test"]["rmse"], default=None,
    )
    candidates = promotable or trained
    best_name = min(candidates, key=lambda k: candidates[k]["test"]["rmse"])

    # ---- stochastic-noise guard ------------------------------------------
    # A neural candidate can win a single run on a lucky seed. Measured spread
    # is ~0.9 RMSE, so a win smaller than that is noise, not improvement.
    # Promoting it would also couple the lightweight hourly job to TensorFlow,
    # which it deliberately does not install.
    deterministic = {
        k: v for k, v in candidates.items()
        if not k.startswith(config.STOCHASTIC_PREFIXES)
    }
    noise_note = None
    if best_name.startswith(config.STOCHASTIC_PREFIXES) and deterministic:
        det_best = min(deterministic, key=lambda k: deterministic[k]["test"]["rmse"])
        margin = deterministic[det_best]["test"]["rmse"] - candidates[best_name]["test"]["rmse"]
        if margin < config.STOCHASTIC_NOISE_RMSE:
            noise_note = (
                f"{best_name} won by {margin:.3f} RMSE over {det_best}, which is "
                f"inside the measured stochastic spread of "
                f"{config.STOCHASTIC_NOISE_RMSE}. Treating it as noise and "
                f"promoting the deterministic model instead."
            )
            log.warning("%s", noise_note)
            best_name = det_best

    best_metrics = results[best_name]["test"]

    beats_baseline = baseline_rmse is None or best_metrics["rmse"] < baseline_rmse
    improvement = (
        None if baseline_rmse in (None, 0)
        else round(100 * (baseline_rmse - best_metrics["rmse"]) / baseline_rmse, 1)
    )
    # Promotion gate. A model that cannot beat "tomorrow looks like today" has
    # learned nothing useful, and shipping it silently replaces a working
    # forecast with a worse one. Refuse the promotion and keep whatever is
    # already in production, unless explicitly overridden.
    if not beats_baseline:
        log.error(
            "best model (%s, RMSE %.2f) does not beat persistence (%.2f)",
            best_name, best_metrics["rmse"], baseline_rmse,
        )
        if not force:
            existing = get_model_registry().production_path()
            if existing is not None:
                raise ModelRejected(
                    f"{best_name} (RMSE {best_metrics['rmse']:.2f}) failed to beat the "
                    f"persistence baseline (RMSE {baseline_rmse:.2f}); keeping the "
                    f"current production model. Re-run with force=True to override."
                )
            log.warning("no existing production model -- promoting anyway so the "
                        "system has something to serve")

    # --------------------------------------------------------------- shap ---
    shap_summary = None
    if not skip_shap:
        try:
            shap_summary = compute_shap(
                fitted[best_name], X_test, feat_cols, model_name=best_name
            )
        except Exception as exc:
            log.warning("SHAP computation failed: %s", exc)

    # ----------------------------------------------------------- register ---
    metadata = {
        "best_model": best_name,
        "beats_persistence_baseline": bool(beats_baseline),
        "stochastic_guard": noise_note,
        "requires_tensorflow": best_name.startswith(config.STOCHASTIC_PREFIXES),
        "improvement_over_baseline_pct": improvement,
        "metrics": {
            "rmse": best_metrics["rmse"],
            "mae": best_metrics["mae"],
            "r2": best_metrics["r2"],
        },
        "metrics_per_horizon": best_metrics["per_horizon"],
        "baseline_rmse": baseline_rmse,
        "sequence_model": _sequence_verdict(trained, seq_winner, best_metrics),
        "all_results": results,
        "feature_columns": feat_cols,
        "target_columns": config.TARGET_COLS,
        "horizons_h": config.HORIZONS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_start": str(train_df["ts"].min()),
        "train_end": str(train_df["ts"].max()),
        "test_start": str(test_df["ts"].min()),
        "test_end": str(test_df["ts"].max()),
        "city": config.CITY,
        "data_source": config.resolved_source(),
        "uses_future_weather": True,
        "future_weather_caveat": (
            "Trained on observed weather shifted backwards (perfect prog). "
            "Serving uses a real Open-Meteo forecast, which carries error, so "
            "live accuracy is expected to sit below these hold-out metrics."
        ),
        "feature_store": config.resolved_store(),
        "trained_at": started.isoformat(),
        "git_sha": _git_sha(),
        "shap_top_features": (shap_summary or {}).get("top_features"),
    }

    staging = Path(config.MODEL_DIR) / f"_staging_{started:%Y%m%d%H%M%S}"
    save_bundle(fitted[best_name], metadata, staging)
    registry = get_model_registry()
    version = registry.push(staging, metadata)
    shutil.rmtree(staging, ignore_errors=True)

    # Comparison table -- consumed by the dashboard's "Model Comparison" tab.
    comparison = [
        {
            "model": name,
            "rmse": r["test"]["rmse"],
            "mae": r["test"]["mae"],
            "r2": r["test"]["r2"],
            "fit_seconds": r["fit_seconds"],
            "rmse_24h": r["test"]["per_horizon"]["24h"]["rmse"],
            "rmse_48h": r["test"]["per_horizon"]["48h"]["rmse"],
            "rmse_72h": r["test"]["per_horizon"]["72h"]["rmse"],
            "is_best": name == best_name,
        }
        for name, r in trained.items()
    ]
    comparison.sort(key=lambda r: r["rmse"])
    (config.ARTIFACT_DIR / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str)
    )

    duration = round((datetime.now(UTC) - started).total_seconds(), 1)
    summary = {
        "version": version,
        "best_model": best_name,
        "test_rmse": round(best_metrics["rmse"], 3),
        "test_mae": round(best_metrics["mae"], 3),
        "test_r2": round(best_metrics["r2"], 4),
        "baseline_rmse": None if baseline_rmse is None else round(baseline_rmse, 3),
        "improvement_over_baseline_pct": improvement,
        "models_trained": len(trained),
        "duration_s": duration,
        "registry": registry.name,
    }
    (config.ARTIFACT_DIR / "last_training_run.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    record_model_metrics(best_name, best_metrics)
    record_pipeline_run("training", duration, ok=True)
    log.info("training complete: %s", summary)
    return summary


def main() -> dict:
    parser = argparse.ArgumentParser(description="Daily AQI training pipeline")
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"subset of: {' '.join(DEFAULT_MODELS + SEQUENCE_MODELS)}")
    parser.add_argument("--test-hours", type=int, default=None)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="promote the winner even if it loses to the baseline")
    parser.add_argument("--fast", action="store_true",
                        help="skip the TensorFlow models (quick CI smoke run)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    models = args.models
    if args.fast and not models:
        models = [m for m in DEFAULT_MODELS if not m.startswith("tf_")]

    summary = run(models=models, test_hours=args.test_hours,
                  skip_shap=args.skip_shap, force=args.force)
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
