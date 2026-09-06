"""One-command bootstrap: backfill -> train -> EDA -> forecast.

    python scripts/bootstrap.py                # full run, 365 days
    python scripts/bootstrap.py --days 90 --fast   # quick demo run

Use this on a fresh clone or before a demo. Every step is idempotent, so it is
safe to re-run.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

log = logging.getLogger("bootstrap")


def step(number: int, total: int, title: str):
    print(f"\n{'=' * 68}\n[{number}/{total}] {title}\n{'=' * 68}")
    return time.perf_counter()


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the AQI Predictor")
    parser.add_argument("--days", type=int, default=config.BACKFILL_DAYS)
    parser.add_argument("--fast", action="store_true",
                        help="skip the TensorFlow models (much quicker)")
    parser.add_argument("--skip-backfill", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"Configuration: {config.summary()}")

    total = 4
    n = 0

    if not args.skip_backfill:
        n += 1
        t0 = step(n, total, f"Backfilling {args.days} days of history")
        from src.pipelines.backfill import run as backfill

        result = backfill(days=args.days)
        print(f"  {result['feature_rows']:,} feature rows "
              f"({result['coverage_start'][:10]} -> {result['coverage_end'][:10]}) "
              f"in {time.perf_counter() - t0:.0f}s")
    else:
        total -= 1

    n += 1
    t0 = step(n, total, "Training and comparing models")
    from src.pipelines.training_pipeline import DEFAULT_MODELS
    from src.pipelines.training_pipeline import run as train

    models = [m for m in DEFAULT_MODELS if not m.startswith("tf_")] if args.fast else None
    result = train(models=models)
    print(f"  best: {result['best_model']}  RMSE {result['test_rmse']}  "
          f"MAE {result['test_mae']}  R2 {result['test_r2']}  "
          f"({time.perf_counter() - t0:.0f}s)")

    n += 1
    t0 = step(n, total, "Generating EDA report and figures")
    try:
        from src.eda import run as run_eda

        eda_result = run_eda()
        print(f"  {len(eda_result['figures'])} figures -> reports/figures/")
        print("  report -> reports/eda_report.md")
    except Exception as exc:
        print(f"  EDA skipped: {exc}")

    n += 1
    t0 = step(n, total, "Producing the 3-day forecast")
    from src.pipelines.inference import clear_cache, predict

    clear_cache()
    forecast = predict()
    print(f"  Current AQI {forecast['current']['aqi']} ({forecast['current']['category']})")
    for f in forecast["forecast"]:
        print(f"  +{f['horizon_hours']:>2}h  AQI {f['aqi']:>6.1f}  "
              f"[{f['aqi_lower']:.0f}-{f['aqi_upper']:.0f}]  {f['category']}")
    if forecast["alerts"]:
        print(f"  {len(forecast['alerts'])} ALERT(S) RAISED")

    print(f"\n{'=' * 68}\nDone. Next:\n"
          f"  streamlit run app/streamlit_app.py\n"
          f"  python -m api.flask_api\n{'=' * 68}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
