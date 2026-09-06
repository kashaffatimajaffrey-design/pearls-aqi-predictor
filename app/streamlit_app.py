"""Streamlit dashboard -- the front end of the AQI Predictor.

Six tabs:
  Forecast        the 3-day prediction, current conditions, active alerts
  History         observed AQI and pollutant series from the Feature Store
  EDA             the exploratory analysis, computed live from stored features
  Explainability  global SHAP importance + a local per-prediction breakdown
  Models          the comparison table from the most recent training run
  Pipeline        freshness, run history and system health

Everything reads through the Feature Store and Model Registry interfaces, so the
same app runs against local Parquet or against Hopsworks with no code change.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, eda  # noqa: E402
from src.aqi import CATEGORIES, advice, categorize  # noqa: E402
from src.data import fetch_aqicn_current  # noqa: E402
from src.explain import explain_prediction, load_shap_summary  # noqa: E402
from src.pipelines import inference  # noqa: E402
from src.store import get_feature_store, get_model_registry  # noqa: E402

st.set_page_config(
    page_title=f"AQI Predictor -- {config.CITY}",
    page_icon="\N{DASH SYMBOL}",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  .aqi-hero {border-radius: 14px; padding: 1.4rem 1.6rem; color: #10131a;
             box-shadow: 0 2px 12px rgba(0,0,0,.10);}
  .aqi-hero h1 {margin: 0; font-size: 3.4rem; line-height: 1;}
  .aqi-hero p  {margin: .25rem 0 0; font-weight: 600;}
  .aqi-card {border-radius: 12px; padding: 1rem 1.1rem; color: #10131a; height: 100%;}
  .aqi-card .val {font-size: 2.1rem; font-weight: 700; line-height: 1.1;}
  .aqi-card .lbl {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
                  opacity: .75;}
  .muted {color: #6b7280; font-size: .85rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------ data access ---
@st.cache_data(ttl=600, show_spinner=False)
def load_forecast(live: bool = False) -> dict:
    return inference.predict(live=live)


@st.cache_data(ttl=600, show_spinner=False)
def load_features() -> pd.DataFrame:
    return get_feature_store().read()


@st.cache_data(ttl=600, show_spinner=False)
def load_backtest(days: int) -> pd.DataFrame:
    try:
        return inference.backtest(days=days)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_aqicn() -> dict | None:
    return fetch_aqicn_current()


def read_artifact(name: str):
    path = config.ARTIFACT_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def refresh_all():
    st.cache_data.clear()
    inference.clear_cache()


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.title("\N{DASH SYMBOL} AQI Predictor")
    st.caption("3-day air-quality forecasting on a serverless stack")

    if st.button("Refresh now", width="stretch", type="primary"):
        refresh_all()
        st.rerun()

    live_mode = st.toggle(
        "Live fetch", value=False,
        help="Bypass the Feature Store and pull straight from the upstream API.",
    )

    st.divider()
    st.subheader("Configuration")
    cfg = config.summary()
    st.write(f"**City** {cfg['city']}, {config.COUNTRY}")
    st.write(f"**Coords** {cfg['lat']:.4f}, {cfg['lon']:.4f}")
    st.write(f"**Source** `{cfg['data_source']}`")
    st.write(f"**Store** `{cfg['feature_store']}`")
    st.write(f"**Alert at** AQI {cfg['alert_threshold']}")

    st.divider()
    st.subheader("AQI scale")
    for lo, hi, name, color in CATEGORIES:
        st.markdown(
            f"<div style='background:{color};padding:.3rem .55rem;border-radius:6px;"
            f"margin-bottom:.25rem;color:#10131a;font-size:.78rem;font-weight:600'>"
            f"{lo}-{hi} &nbsp;{name}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Built for the 10Pearls internship programme.")


# ------------------------------------------------------------------ load ----
try:
    forecast = load_forecast(live=live_mode)
except Exception as exc:
    st.error(f"Could not produce a forecast: {exc}")
    st.info(
        "First run? Bootstrap the project with:\n\n"
        "```bash\n"
        "python -m src.pipelines.backfill --days 365\n"
        "python -m src.pipelines.training_pipeline\n"
        "```"
    )
    st.stop()

features = load_features()
current = forecast["current"]

st.title(f"Air Quality Forecast -- {forecast['city']}")
st.caption(
    f"Observation at {forecast['as_of'][:16].replace('T', ' ')} UTC \N{BULLET} "
    f"forecast generated {forecast['generated_at'][:16].replace('T', ' ')} UTC \N{BULLET} "
    f"model `{forecast['model']['name']}` {forecast['model'].get('version') or ''}"
)

# --------------------------------------------------------------- alerts -----
for alert in forecast.get("alerts") or []:
    st.error(f"**{alert['message']}** -- {alert['advice']}", icon="\N{WARNING SIGN}")
if not forecast.get("alerts"):
    st.success(
        f"No AQI above the alert threshold ({config.ALERT_AQI_THRESHOLD}) "
        "in the next 72 hours.",
        icon="\N{WHITE HEAVY CHECK MARK}",
    )

tabs = st.tabs(
    ["Forecast", "History", "EDA", "Explainability", "Models", "Pipeline"]
)


# =============================================================== FORECAST ====
with tabs[0]:
    hero, *cards = st.columns([1.5, 1, 1, 1])

    with hero:
        st.markdown(
            f"<div class='aqi-hero' style='background:{current['color']}'>"
            f"<div class='lbl'>Current AQI</div>"
            f"<h1>{current['aqi']:.0f}</h1>"
            f"<p>{current['category']}</p></div>",
            unsafe_allow_html=True,
        )
        st.caption(advice(current["aqi"]))

    for col, f in zip(cards, forecast["forecast"], strict=False):
        with col:
            delta = f["aqi"] - current["aqi"]
            st.markdown(
                f"<div class='aqi-card' style='background:{f['color']}'>"
                f"<div class='lbl'>In {f['horizon_days']} day"
                f"{'s' if f['horizon_days'] > 1 else ''}</div>"
                f"<div class='val'>{f['aqi']:.0f}</div>"
                f"<div style='font-size:.8rem;font-weight:600'>{f['category']}</div>"
                f"<div style='font-size:.72rem;opacity:.8'>"
                f"{delta:+.0f} vs now &nbsp;\N{BULLET}&nbsp; "
                f"{f['aqi_lower']:.0f}-{f['aqi_upper']:.0f}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Forecast trajectory")

    hist = features.tail(24 * 7) if not features.empty else pd.DataFrame()
    fig = go.Figure()

    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["ts"], y=hist["aqi"], name="Observed",
            line=dict(color="#1f2937", width=2),
        ))

    as_of = pd.Timestamp(forecast["as_of"])
    fx = [as_of] + [pd.Timestamp(f["valid_at"]) for f in forecast["forecast"]]
    fy = [current["aqi"]] + [f["aqi"] for f in forecast["forecast"]]
    lo = [current["aqi"]] + [f["aqi_lower"] for f in forecast["forecast"]]
    hi = [current["aqi"]] + [f["aqi_upper"] for f in forecast["forecast"]]

    fig.add_trace(go.Scatter(x=fx + fx[::-1], y=hi + lo[::-1], fill="toself",
                             fillcolor="rgba(231,76,60,.15)", line=dict(width=0),
                             name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=fy, name="Forecast", mode="lines+markers",
                             line=dict(color="#e74c3c", width=3, dash="dash"),
                             marker=dict(size=10)))

    for lo_b, hi_b, _name, color in CATEGORIES[:5]:
        fig.add_hrect(y0=lo_b, y1=hi_b, fillcolor=color, opacity=0.08,
                      line_width=0, layer="below")
    fig.add_hline(y=config.ALERT_AQI_THRESHOLD, line_dash="dot", line_color="crimson",
                  annotation_text="Alert threshold")
    fig.add_vline(x=as_of.timestamp() * 1000, line_dash="dot", line_color="#555",
                  annotation_text="now")

    fig.update_layout(height=430, hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="AQI", xaxis_title=None,
                      legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Current conditions")
    m = st.columns(6)
    m[0].metric("PM2.5", f"{current.get('pm2_5') or 0:.1f} \N{MICRO SIGN}g/m\N{SUPERSCRIPT THREE}")
    m[1].metric("PM10", f"{current.get('pm10') or 0:.1f} \N{MICRO SIGN}g/m\N{SUPERSCRIPT THREE}")
    m[2].metric("Temperature", f"{current.get('temperature') or 0:.1f} \N{DEGREE SIGN}C")
    m[3].metric("Humidity", f"{current.get('humidity') or 0:.0f} %")
    m[4].metric("Wind", f"{current.get('wind_speed') or 0:.1f} m/s")
    m[5].metric("Dominant", str(current.get("dominant_pollutant", "n/a")).upper())

    reference = load_aqicn()
    if reference and reference.get("aqi") is not None:
        st.caption(
            f"AQICN cross-check \N{EM DASH} nearest station "
            f"**{reference.get('station', 'n/a')}** reports AQI "
            f"**{reference['aqi']}** (dominant `{reference.get('dominant_pollutant')}`)."
        )

    with st.expander("Forecast detail"):
        st.dataframe(
            pd.DataFrame(forecast["forecast"])[
                ["horizon_hours", "valid_at", "aqi", "aqi_lower", "aqi_upper",
                 "category", "model_rmse", "advice"]
            ],
            width="stretch", hide_index=True,
        )


# ================================================================ HISTORY ====
with tabs[1]:
    if features.empty:
        st.warning("No history in the Feature Store yet. Run the backfill.")
    else:
        window = st.select_slider(
            "Window", options=[1, 3, 7, 14, 30, 90, 180, 365], value=30,
            format_func=lambda d: f"{d} day{'s' if d > 1 else ''}",
        )
        hist = features[features["ts"] >= features["ts"].max() - timedelta(days=window)]

        c = st.columns(4)
        c[0].metric("Mean AQI", f"{hist['aqi'].mean():.1f}")
        c[1].metric("Peak AQI", f"{hist['aqi'].max():.0f}")
        c[2].metric("Hours above threshold",
                    int((hist["aqi"] >= config.ALERT_AQI_THRESHOLD).sum()))
        c[3].metric("Observations", f"{len(hist):,}")

        fig = px.line(hist, x="ts", y="aqi", title=f"AQI, last {window} days")
        for lo_b, hi_b, _name, color in CATEGORIES[:5]:
            fig.add_hrect(y0=lo_b, y1=hi_b, fillcolor=color, opacity=0.08,
                          line_width=0, layer="below")
        fig.update_traces(line_color="#1f2937")
        fig.update_layout(height=380, yaxis_title="AQI", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")

        pollutants = [p for p in config.POLLUTANTS if p in hist.columns]
        chosen = st.multiselect("Pollutants", pollutants,
                                default=[p for p in ("pm2_5", "pm10") if p in pollutants])
        if chosen:
            st.plotly_chart(
                px.line(hist, x="ts", y=chosen,
                        labels={"value": "\N{MICRO SIGN}g/m\N{SUPERSCRIPT THREE}", "ts": ""})
                .update_layout(height=340),
                width="stretch",
            )

        st.markdown("### Predicted vs actual (backtest)")
        bt = load_backtest(min(window, 60))
        if bt.empty:
            st.info("Backtest needs a trained model and enough history.")
        else:
            horizon = st.radio("Horizon", config.HORIZONS, horizontal=True,
                               format_func=lambda h: f"+{h}h")
            sub = bt[bt["horizon_hours"] == horizon]
            if sub.empty:
                st.info("No overlapping actuals for this horizon yet.")
            else:
                k = st.columns(3)
                k[0].metric("MAE", f"{sub['abs_error'].mean():.2f}")
                k[1].metric("RMSE", f"{(sub['error'] ** 2).mean() ** .5:.2f}")
                k[2].metric("Bias", f"{sub['error'].mean():+.2f}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sub["valid_at"], y=sub["actual"],
                                         name="Actual", line=dict(color="#1f2937")))
                fig.add_trace(go.Scatter(x=sub["valid_at"], y=sub["predicted"],
                                         name="Predicted",
                                         line=dict(color="#e74c3c", dash="dash")))
                fig.update_layout(height=360, hovermode="x unified",
                                  yaxis_title="AQI", xaxis_title=None)
                st.plotly_chart(fig, width="stretch")


# ==================================================================== EDA ====
with tabs[2]:
    if features.empty:
        st.warning("No data to analyse yet.")
    else:
        with st.spinner("Computing exploratory statistics..."):
            stats = eda.describe(features)

        c = st.columns(5)
        c[0].metric("Observations", f"{stats['rows']:,}")
        c[1].metric("Span", f"{stats['span_days']:.0f} days")
        c[2].metric("Mean AQI", stats["aqi_mean"])
        c[3].metric("Peak AQI", stats["aqi_max"])
        c[4].metric("Unhealthy hours", f"{stats['unhealthy_pct']}%")

        st.markdown(
            f"AQI peaks around **{stats['peak_hour']:02d}:00 UTC** "
            f"({stats['peak_hour_aqi']}) and is cleanest at "
            f"**{stats['cleanest_hour']:02d}:00 UTC** ({stats['cleanest_hour_aqi']}). "
            f"The most frequent dominant pollutant is `{stats['dominant_pollutant_mode']}`."
        )

        left, right = st.columns(2)
        with left:
            hourly = features.groupby(features["ts"].dt.hour)["aqi"].mean().reset_index()
            hourly.columns = ["hour", "mean_aqi"]
            st.plotly_chart(
                px.line(hourly, x="hour", y="mean_aqi", markers=True,
                        title="Daily cycle").update_layout(height=310),
                width="stretch",
            )
        with right:
            dow = features.groupby(features["ts"].dt.dayofweek)["aqi"].mean().reset_index()
            dow.columns = ["dow", "mean_aqi"]
            dow["dow"] = dow["dow"].map(
                dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
            )
            st.plotly_chart(
                px.bar(dow, x="dow", y="mean_aqi",
                       title="Weekly cycle").update_layout(height=310),
                width="stretch",
            )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.histogram(features, x="aqi", nbins=60,
                             title="AQI distribution").update_layout(height=330),
                width="stretch",
            )
        with right:
            counts = features["aqi"].dropna().apply(categorize).value_counts()
            order = [c[2] for c in CATEGORIES if c[2] in counts.index]
            st.plotly_chart(
                px.bar(x=order, y=[counts[o] for o in order],
                       color=order, title="Hours per category",
                       color_discrete_map={c[2]: c[3] for c in CATEGORIES},
                       labels={"x": "", "y": "hours"})
                .update_layout(height=330, showlegend=False),
                width="stretch",
            )

        st.markdown("### Autocorrelation")
        ac = eda.autocorrelation(features, 168).reset_index()
        ac.columns = ["lag_hours", "autocorr"]
        fig = px.line(ac, x="lag_hours", y="autocorr",
                      title="How far back AQI carries information")
        for h in config.HORIZONS:
            fig.add_vline(x=h, line_dash="dot", annotation_text=f"+{h}h")
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Correlation stays well above zero out to 72 hours, which is what "
            "justifies the 1-72h lag window in the feature set -- and also marks "
            "the honest limit of the forecast."
        )

        st.markdown("### Correlation matrix")
        st.plotly_chart(
            px.imshow(eda.correlations(features), text_auto=".2f",
                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                      aspect="auto").update_layout(height=520),
            width="stretch",
        )

        report = config.REPORT_DIR / "eda_report.md"
        if report.exists():
            with st.expander("Full written EDA report"):
                st.markdown(report.read_text(encoding="utf-8"))


# ========================================================= EXPLAINABILITY ====
with tabs[3]:
    st.markdown("### Global feature importance (SHAP)")
    shap_summary = load_shap_summary()

    if shap_summary is None:
        st.info("No SHAP summary yet. It is produced by the training pipeline.")
    else:
        st.caption(
            f"`{shap_summary['explainer']}` over {shap_summary['n_rows_explained']} "
            f"hold-out rows, model `{shap_summary['model']}`, "
            f"horizon +{shap_summary['horizon_hours']}h."
        )
        imp = pd.DataFrame(shap_summary["top_features"]).head(18)
        fig = px.bar(
            imp.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
            orientation="h", color="direction",
            color_discrete_map={"increases_aqi": "#c0392b",
                                "decreases_aqi": "#2980b9",
                                "neutral": "#95a5a6"},
            labels={"mean_abs_shap": "mean |SHAP|", "feature": ""},
        )
        fig.update_layout(height=560, legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Bar length is how much a feature moves the prediction on average; "
            "colour is the direction of that push."
        )

    st.divider()
    st.markdown("### Why this specific forecast?")
    horizon = st.selectbox("Horizon", config.HORIZONS,
                           format_func=lambda h: f"+{h} hours")

    if st.button("Explain this prediction"):
        with st.spinner("Running SHAP on the latest feature row..."):
            try:
                pipeline, metadata = inference._load_production()
                cols = metadata.get("feature_columns") or []
                local = explain_prediction(
                    pipeline, features.tail(1), cols,
                    background=features.tail(500),
                    model_name=metadata.get("best_model", ""),
                    horizon_index=config.HORIZONS.index(horizon),
                )
                st.write(
                    f"Base value **{local['base_value']}** \N{RIGHTWARDS ARROW} "
                    f"prediction **{local['prediction']}**"
                )
                contrib = pd.DataFrame(local["contributions"])
                fig = px.bar(
                    contrib.sort_values("shap"), x="shap", y="feature",
                    orientation="h", color="shap",
                    color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                    labels={"shap": "contribution to AQI", "feature": ""},
                )
                fig.update_layout(height=430, coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")
                st.dataframe(contrib, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Local explanation failed: {exc}")


# ================================================================= MODELS ====
with tabs[4]:
    comparison = read_artifact("model_comparison.json")
    if not comparison:
        st.info("No comparison yet. Run the training pipeline.")
    else:
        st.markdown("### Candidate comparison (hold-out test set)")
        df = pd.DataFrame(comparison)
        best = df[df["is_best"]].iloc[0] if df["is_best"].any() else df.iloc[0]
        st.success(
            f"Production model: **{best['model']}** \N{EM DASH} "
            f"RMSE {best['rmse']:.2f}, MAE {best['mae']:.2f}, R\N{SUPERSCRIPT TWO} {best['r2']:.3f}"
        )
        st.dataframe(
            df[["model", "rmse", "mae", "r2", "rmse_24h", "rmse_48h", "rmse_72h",
                "fit_seconds", "is_best"]]
            .style.format({"rmse": "{:.2f}", "mae": "{:.2f}", "r2": "{:.3f}",
                           "rmse_24h": "{:.2f}", "rmse_48h": "{:.2f}",
                           "rmse_72h": "{:.2f}", "fit_seconds": "{:.1f}"})
            .background_gradient(subset=["rmse"], cmap="RdYlGn_r"),
            width="stretch", hide_index=True,
        )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.bar(df.sort_values("rmse"), x="model", y="rmse",
                       color="is_best", title="RMSE by model",
                       color_discrete_map={True: "#27ae60", False: "#95a5a6"})
                .update_layout(height=360, showlegend=False),
                width="stretch",
            )
        with right:
            # value_name must not collide with the existing "rmse" column.
            long = df.melt(
                id_vars="model", value_vars=["rmse_24h", "rmse_48h", "rmse_72h"],
                var_name="horizon", value_name="horizon_rmse",
            )
            long["horizon"] = long["horizon"].str.replace("rmse_", "+")
            st.plotly_chart(
                px.line(long, x="horizon", y="horizon_rmse", color="model", markers=True,
                        title="Error growth with horizon",
                        labels={"horizon_rmse": "RMSE"}).update_layout(height=360),
                width="stretch",
            )
        st.caption(
            "Error grows with horizon for every model \N{EM DASH} that is the physics of "
            "the problem, not a bug. The persistence baseline is included so you "
            "can see how much the learned models actually add."
        )

        history = get_model_registry().history()[:10]
        if history:
            st.markdown("### Registry history")
            st.dataframe(
                pd.DataFrame([
                    {"version": h.get("version"), "model": h.get("best_model"),
                     "rmse": (h.get("metrics") or {}).get("rmse"),
                     "r2": (h.get("metrics") or {}).get("r2"),
                     "trained_at": h.get("trained_at"), "git_sha": h.get("git_sha")}
                    for h in history
                ]),
                width="stretch", hide_index=True,
            )


# =============================================================== PIPELINE ====
with tabs[5]:
    st.markdown("### System health")

    feature_run = read_artifact("last_feature_run.json")
    training_run = read_artifact("last_training_run.json")
    backfill = read_artifact("last_backfill.json")

    c = st.columns(4)
    if not features.empty:
        age = (datetime.now(UTC) - features["ts"].max().to_pydatetime())
        hours = age.total_seconds() / 3600
        c[0].metric("Feature freshness", f"{hours:.1f} h",
                    delta="fresh" if hours < 6 else "stale",
                    delta_color="normal" if hours < 6 else "inverse")
        c[1].metric("Rows in store", f"{len(features):,}")
    if training_run:
        c[2].metric("Model version", training_run.get("version", "n/a"))
        c[3].metric("Test RMSE", training_run.get("test_rmse", "n/a"))

    left, right = st.columns(2)
    with left:
        st.markdown("**Last feature pipeline run**")
        st.json(feature_run or {"status": "no run recorded yet"})
        st.markdown("**Last backfill**")
        st.json(backfill or {"status": "no backfill recorded yet"})
    with right:
        st.markdown("**Last training run**")
        st.json(training_run or {"status": "no run recorded yet"})
        st.markdown("**Monitoring state**")
        st.json(read_artifact("monitoring_state.json") or {})

    st.divider()
    st.markdown("### Automation")
    st.markdown(
        """
| Job | Schedule | Runner |
| --- | --- | --- |
| `feature_pipeline.yml` | hourly (`0 * * * *`) | GitHub Actions |
| `training_pipeline.yml` | daily (`30 2 * * *`) | GitHub Actions |
| `ci.yml` | every push / PR | GitHub Actions |

Each run commits the refreshed Parquet, model bundle and artifacts back to the
repository, which is what makes the dashboard update with no manual step.
Prometheus scrapes `/metrics` on the Flask API; Grafana reads from Prometheus.
"""
    )
