"""Model explainability with SHAP (global + local) and LIME (local).

Two audiences:

* Global -- "what drives AQI in this city?" Mean |SHAP| over the test set,
  computed once during training and cached to disk so the dashboard renders it
  without re-running the explainer.
* Local  -- "why is tomorrow forecast at 168?" Per-feature contributions for a
  single prediction, computed on demand.

Tree models get the exact TreeExplainer; everything else (Ridge, the Keras nets)
falls back to KernelExplainer over a small K-means background, which is slow but
model-agnostic. LIME is offered alongside SHAP for local explanations because
its linear surrogate is often easier to put in front of a non-technical reader.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)

SHAP_PATH = config.ARTIFACT_DIR / "shap_summary.json"
MAX_BACKGROUND = 100
MAX_EXPLAIN_ROWS = 300

TREE_MODELS = {"random_forest", "gradient_boosting", "xgboost"}


def _split_pipeline(pipeline):
    """Return (preprocessor_fn, final_estimator).

    SHAP must see the matrix the estimator actually consumes, so we push the
    input through the pipeline's transform steps first.
    """
    steps = list(pipeline.named_steps.items())
    estimator = steps[-1][1]

    def transform(X):
        out = X
        for _name, step in steps[:-1]:
            out = step.transform(out)
        return np.asarray(out, dtype=float)

    return transform, estimator


def _unwrap_multioutput(estimator, horizon_index: int):
    """MultiOutputRegressor holds one estimator per target; pick the one for the
    horizon we are explaining."""
    if hasattr(estimator, "estimators_"):
        return estimator.estimators_[horizon_index]
    return estimator


def compute_shap(pipeline, X: pd.DataFrame, feature_names: list[str],
                 model_name: str = "", horizon_index: int = 0,
                 max_rows: int = MAX_EXPLAIN_ROWS) -> dict:
    """Global SHAP summary for the +24h horizon. Cached to shap_summary.json."""
    import shap

    X = X.tail(max_rows)
    transform, estimator = _split_pipeline(pipeline)
    X_t = transform(X)

    target = _unwrap_multioutput(estimator, horizon_index)

    if model_name in TREE_MODELS or hasattr(target, "feature_importances_"):
        explainer = shap.TreeExplainer(target)
        values = explainer.shap_values(X_t)
    else:
        background = shap.kmeans(X_t, min(20, len(X_t)))
        # Wrap the multi-output predict down to the single horizon we explain.
        def _predict(arr):
            out = np.asarray(estimator.predict(arr))
            return out[:, horizon_index] if out.ndim > 1 else out

        explainer = shap.KernelExplainer(_predict, background)
        subset = X_t[: min(60, len(X_t))]  # KernelExplainer is O(rows * features)
        values = explainer.shap_values(subset, nsamples=100, silent=True)

    values = np.asarray(values)
    if values.ndim == 3:                      # (rows, features, outputs)
        values = values[:, :, horizon_index]

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    top_features = [
        {
            "feature": feature_names[i],
            "mean_abs_shap": float(mean_abs[i]),
            # Sign of the correlation between feature value and its SHAP value:
            # tells you whether "more of this" pushes AQI up or down.
            "direction": _direction(X_t[:len(values), i], values[:, i]),
        }
        for i in order[:25]
    ]

    summary = {
        "model": model_name,
        "horizon_hours": config.HORIZONS[horizon_index],
        "n_rows_explained": int(len(values)),
        "explainer": "TreeExplainer" if model_name in TREE_MODELS else "KernelExplainer",
        "top_features": top_features,
    }
    SHAP_PATH.write_text(json.dumps(summary, indent=2, default=str))
    log.info("SHAP summary written (%d features ranked)", len(top_features))
    return summary


def _direction(feature_values: np.ndarray, shap_values: np.ndarray) -> str:
    if len(feature_values) < 3 or np.std(feature_values) == 0:
        return "neutral"
    corr = np.corrcoef(feature_values, shap_values)[0, 1]
    if np.isnan(corr) or abs(corr) < 0.1:
        return "neutral"
    return "increases_aqi" if corr > 0 else "decreases_aqi"


def load_shap_summary() -> dict | None:
    if not SHAP_PATH.exists():
        return None
    try:
        return json.loads(SHAP_PATH.read_text())
    except json.JSONDecodeError:
        return None


def explain_prediction(pipeline, row: pd.DataFrame, feature_names: list[str],
                       background: pd.DataFrame | None = None,
                       model_name: str = "", horizon_index: int = 0,
                       top_n: int = 12) -> dict:
    """Local SHAP: why this particular forecast came out where it did."""
    import shap

    transform, estimator = _split_pipeline(pipeline)
    x_t = transform(row[feature_names])
    target = _unwrap_multioutput(estimator, horizon_index)

    if model_name in TREE_MODELS or hasattr(target, "feature_importances_"):
        explainer = shap.TreeExplainer(target)
        values = np.asarray(explainer.shap_values(x_t))
        base = explainer.expected_value
    else:
        if background is None or background.empty:
            raise ValueError("a background sample is required for non-tree models")
        bg_t = transform(background[feature_names].tail(MAX_BACKGROUND))

        def _predict(arr):
            out = np.asarray(estimator.predict(arr))
            return out[:, horizon_index] if out.ndim > 1 else out

        explainer = shap.KernelExplainer(_predict, shap.kmeans(bg_t, 15))
        values = np.asarray(explainer.shap_values(x_t, nsamples=100, silent=True))
        base = explainer.expected_value

    if values.ndim == 3:
        values = values[:, :, horizon_index]
    values = values.ravel()
    base = float(np.ravel(base)[horizon_index] if np.ndim(base) else base)

    order = np.argsort(np.abs(values))[::-1][:top_n]
    contributions = [
        {
            "feature": feature_names[i],
            "value": _fmt(row[feature_names].iloc[0, i]),
            "shap": round(float(values[i]), 3),
            "effect": "pushes AQI up" if values[i] > 0 else "pushes AQI down",
        }
        for i in order
    ]

    return {
        "horizon_hours": config.HORIZONS[horizon_index],
        "base_value": round(base, 2),
        "prediction": round(base + float(values.sum()), 2),
        "contributions": contributions,
    }


def lime_explain(pipeline, row: pd.DataFrame, background: pd.DataFrame,
                 feature_names: list[str], horizon_index: int = 0,
                 top_n: int = 10) -> dict:
    """Local LIME explanation -- a linear surrogate fitted around this one point."""
    from lime.lime_tabular import LimeTabularExplainer

    bg = background[feature_names].tail(2000).fillna(background[feature_names].median())
    explainer = LimeTabularExplainer(
        bg.to_numpy(dtype=float),
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True,
        random_state=config.RANDOM_STATE,
    )

    def _predict(arr):
        frame = pd.DataFrame(arr, columns=feature_names)
        out = np.asarray(pipeline.predict(frame))
        return out[:, horizon_index] if out.ndim > 1 else out

    x = row[feature_names].fillna(bg.median()).to_numpy(dtype=float)[0]
    exp = explainer.explain_instance(x, _predict, num_features=top_n)

    return {
        "horizon_hours": config.HORIZONS[horizon_index],
        "intercept": round(float(exp.intercept[0]), 3),
        "local_r2": round(float(exp.score), 3),
        "contributions": [
            {"rule": rule, "weight": round(float(weight), 3),
             "effect": "pushes AQI up" if weight > 0 else "pushes AQI down"}
            for rule, weight in exp.as_list()
        ],
    }


def _fmt(value):
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return str(value)
