"""Measure the ceiling on 'forecast weather as known-future covariates'.

Trick: we already hold a full year of OBSERVED weather. Shifting it backwards
(-24/-48/-72h) gives the model a *perfect* weather forecast -- so this measures
the UPPER BOUND of the improvement. A real Open-Meteo forecast carries error, so
the realised gain will be some fraction of whatever this shows.

No new API calls needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.features import build_targets, feature_columns
from src.models import build_model
from src.pipelines.training_pipeline import _score, chronological_split
from src.store import get_feature_store

FUTURE_WX = ["temperature", "humidity", "pressure", "wind_speed",
             "precipitation", "boundary_layer_height", "wind_u", "wind_v",
             "ventilation_index"]


def add_future_weather(df):
    out = df.copy()
    added = []
    for h in config.HORIZONS:
        for col in FUTURE_WX:
            if col in out.columns:
                name = f"{col}_fut_{h}h"
                out[name] = out[col].shift(-h)
                added.append(name)
    return out, added


def run():
    df = get_feature_store().read()
    print(f"loaded {len(df):,} rows")

    # --- control: exactly the current feature set ---
    lab = build_targets(df).dropna(subset=config.TARGET_COLS)
    base_cols = feature_columns(lab)

    # --- treatment: + perfect-knowledge future weather ---
    aug, added = add_future_weather(df)
    aug_lab = build_targets(aug).dropna(subset=config.TARGET_COLS)
    aug_cols = feature_columns(aug_lab)

    print(f"control features : {len(base_cols)}")
    print(f"treatment features: {len(aug_cols)}  (+{len(added)} future-weather)")

    results = {}
    for label, frame, cols in (("control", lab, base_cols),
                               ("future_weather", aug_lab, aug_cols)):
        tr, te = chronological_split(frame, config.TEST_SIZE_HOURS)
        model = build_model("xgboost")
        model.fit(tr[cols], tr[config.TARGET_COLS].to_numpy(float))
        s = _score(te[config.TARGET_COLS].to_numpy(float), model.predict(te[cols]))
        results[label] = s
        print(f"\n{label:16} RMSE {s['rmse']:.3f}  MAE {s['mae']:.3f}  R2 {s['r2']:.4f}")
        for h in config.HORIZONS:
            ph = s["per_horizon"][f"{h}h"]
            print(f"    +{h:>2}h  RMSE {ph['rmse']:6.3f}  MAE {ph['mae']:6.3f}  R2 {ph['r2']:7.4f}")

    c, t = results["control"], results["future_weather"]
    print("\n" + "=" * 62)
    print("CEILING ON IMPROVEMENT (perfect weather forecast)")
    print("=" * 62)
    print(f"overall RMSE {c['rmse']:.3f} -> {t['rmse']:.3f}  "
          f"({100*(c['rmse']-t['rmse'])/c['rmse']:+.1f}%)")
    print(f"overall R2   {c['r2']:.4f} -> {t['r2']:.4f}")
    for h in config.HORIZONS:
        cr = c["per_horizon"][f"{h}h"]["rmse"]
        tr_ = t["per_horizon"][f"{h}h"]["rmse"]
        print(f"  +{h:>2}h  RMSE {cr:6.3f} -> {tr_:6.3f}  ({100*(cr-tr_)/cr:+5.1f}%)")


if __name__ == "__main__":
    run()
