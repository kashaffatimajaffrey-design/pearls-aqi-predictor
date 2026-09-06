"""Is the encoder-decoder LSTM actually better, or was that one lucky run?

A single training run produced RMSE 8.50 for the future-weather variant and 9.80
without it -- an apparently decisive win. A second run of identical code gave
9.72 and 9.62, reversing the ordering.

Neural training here is stochastic (weight init, dropout, CPU reduction order),
so a single-run comparison is not evidence. This repeats both variants across
seeds and reports mean +/- std, which is the only fair way to compare them
against a deterministic XGBoost.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src import config  # noqa: E402
from src.features import build_targets  # noqa: E402
from src.models.sequence import SequenceForecaster, build_sequences  # noqa: E402
from src.pipelines.training_pipeline import _score, chronological_split  # noqa: E402
from src.store import get_feature_store  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def run():
    import pandas as pd

    df = get_feature_store().read()
    labelled = build_targets(df).dropna(subset=config.TARGET_COLS)
    _, test_df = chronological_split(labelled, config.TEST_SIZE_HOURS)
    split_ts = pd.Timestamp(test_df["ts"].min())

    out = {}
    for use_future in (True, False):
        label = "with_future_weather" if use_future else "no_future_weather"
        Xp, Xf, y, ts, _, _ = build_sequences(
            labelled, config.TARGET_COLS, use_future=use_future
        )
        mask = np.asarray(ts < split_ts)
        scores = []
        for seed in SEEDS:
            m = SequenceForecaster(use_future=use_future, seed=seed)
            m.fit(Xp[mask], Xf[mask], y[mask])
            s = _score(y[~mask], m.predict(Xp[~mask], Xf[~mask]))
            scores.append(s["rmse"])
            print(f"  {label:22} seed={seed}  RMSE {s['rmse']:.3f}  R2 {s['r2']:+.3f}")
        out[label] = np.array(scores)

    print("\n" + "=" * 64)
    print(f"{'variant':24} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}")
    print("=" * 64)
    for label, arr in out.items():
        print(f"{label:24} {arr.mean():7.3f} {arr.std():7.3f} {arr.min():7.3f} {arr.max():7.3f}")
    print(f"{'xgboost (deterministic)':24} {8.628:7.3f} {0.0:7.3f} {8.628:7.3f} {8.628:7.3f}")

    a, b = out["with_future_weather"], out["no_future_weather"]
    print(f"\nfuture-weather effect: {b.mean() - a.mean():+.3f} RMSE "
          f"(pooled std {np.sqrt((a.std()**2 + b.std()**2) / 2):.3f})")
    print(f"best sequence mean {min(a.mean(), b.mean()):.3f} vs xgboost 8.628 -> "
          f"{'sequence wins' if min(a.mean(), b.mean()) < 8.628 else 'XGBOOST STILL WINS'}")


if __name__ == "__main__":
    run()
