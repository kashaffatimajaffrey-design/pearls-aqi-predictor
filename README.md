# Pearls AQI Predictor

Three-day Air Quality Index forecasting on a fully serverless stack, built for
the 10Pearls internship programme.

The system ingests hourly pollutant and weather observations, engineers ~150
features, trains and compares seven model families, promotes the best one to a
model registry, and serves the forecast through a Streamlit dashboard and a
Flask REST API — with SHAP explanations, hazardous-AQI alerting and
Prometheus/Grafana monitoring. Every stage runs on a schedule in GitHub Actions,
so the dataset, the model and the dashboard update on their own with no manual
step.

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                              # optional — every value has a default
```

Bootstrap the project end to end:

```bash
python -m src.pipelines.backfill --days 365
```

```bash
python -m src.pipelines.training_pipeline
```

```bash
streamlit run app/streamlit_app.py
```

The dashboard opens on <http://localhost:8501>. The Flask API runs separately:

```bash
python -m api.flask_api
```

**No API keys are required.** The default data source is Open-Meteo, which is
keyless. Setting `OPENWEATHER_API_KEY` switches the pipeline to OpenWeather;
setting `HOPSWORKS_API_KEY` + `HOPSWORKS_PROJECT` switches the feature store and
model registry to Hopsworks. Both are single environment-variable changes — no
code edits.

---

## Architecture

```
                          ┌──────────────────────────────────┐
   OpenWeather / AQICN ──▶│  Feature Pipeline    (hourly)    │
   Open-Meteo (keyless)   │  fetch → EPA AQI → 150 features  │
                          └────────────────┬─────────────────┘
                                           ▼
                          ┌──────────────────────────────────┐
                          │  Feature Store                   │
                          │  Hopsworks  or  local Parquet    │
                          └────────────────┬─────────────────┘
                                           ▼
                          ┌──────────────────────────────────┐
                          │  Training Pipeline   (daily)     │
                          │  7 models → chronological eval   │
                          │  → best promoted → SHAP          │
                          └────────────────┬─────────────────┘
                                           ▼
                          ┌──────────────────────────────────┐
                          │  Model Registry                  │
                          │  Hopsworks  or  versioned dir    │
                          └────────────────┬─────────────────┘
                                           ▼
        ┌──────────────────────┬───────────┴──────────┬────────────────────┐
        ▼                      ▼                      ▼                    ▼
  Streamlit dashboard    Flask REST API         Alerting              Prometheus
  6 tabs, live           JSON + /metrics        Slack/Discord         → Grafana
```

### Repository layout

| Path | Purpose |
| --- | --- |
| `src/config.py` | Central, environment-overridable configuration |
| `src/aqi.py` | US EPA AQI breakpoints, unit conversion, categories |
| `src/data/sources.py` | OpenWeather / Open-Meteo / AQICN connectors |
| `src/features/engineering.py` | Calendar, lag and derived feature construction |
| `src/store/feature_store.py` | Hopsworks ⇄ local Parquet, one interface |
| `src/store/model_registry.py` | Hopsworks ⇄ versioned directory, one interface |
| `src/models/zoo.py` | The seven candidate models behind one fit/predict API |
| `src/pipelines/` | `backfill`, `feature_pipeline`, `training_pipeline`, `inference` |
| `src/explain/explainer.py` | SHAP (global + local) and LIME |
| `src/monitoring/metrics.py` | Prometheus instrumentation |
| `src/alerts.py` | Hazardous-AQI detection and webhook dispatch |
| `src/eda.py` | EDA figures + generated written report |
| `app/streamlit_app.py` | Dashboard |
| `api/flask_api.py` | REST API |
| `airflow/dags/aqi_dag.py` | Airflow alternative to the GitHub Actions workflows |
| `.github/workflows/` | Hourly, daily and CI automation |
| `monitoring/` | Prometheus config, alert rules, Grafana dashboard |

---

## The forecasting problem

**Target.** AQI at t+24h, t+48h and t+72h — three separate heads, predicted
directly rather than by recursively rolling a one-step model forward. Direct
multi-horizon avoids compounding the one-step error across seventy-two steps.

**AQI computation.** Providers return raw concentrations in µg/m³, so the US EPA
index is computed in `src/aqi.py` rather than taken from an API: gaseous
pollutants are converted to ppb/ppm at 25 °C, averaged over the EPA's own
windows (24 h for particulates, 8 h for O₃/CO), mapped through the piecewise
breakpoint table, and the overall AQI is the maximum sub-index. The 2024 revised
PM2.5 breakpoints are used.

**Features** (~150 columns, all strictly causal):

| Group | Examples |
| --- | --- |
| Calendar | hour, day-of-week, month — each encoded cyclically as sin/cos |
| Lag | AQI, PM2.5, PM10, NO₂, O₃ at t−1, 2, 3, 6, 12, 24, 48, 72 h |
| Rolling | mean / std / min / max over 6, 12, 24, 72 h windows |
| Change rate | Δ1h, Δ3h, Δ24h, per-hour change rate, acceleration, deviation from 24 h mean |
| Weather | temperature, humidity, pressure, precipitation, wind speed |
| Derived weather | wind decomposed into u/v components, `stagnation` = 1/(1+wind), temp×humidity, 24 h temperature range |

Two details worth flagging, because both are places this kind of project usually
goes wrong:

- **Rolling features are shifted by one step** before aggregating, so the current
  value never appears in its own rolling mean. `tests/test_features.py` asserts
  this directly.
- **Wind direction is decomposed into u/v components.** A raw 0–360° column puts
  359 and 1 at opposite ends of the number line despite being adjacent
  physically; trees split on it meaninglessly.

---

## Models compared

| Model | Family | Why it's here |
| --- | --- | --- |
| `baseline_persistence` | statistical | "Tomorrow = today". The bar every learned model must clear |
| `ridge` | linear | L2-regularised; fast, interpretable reference point |
| `random_forest` | bagged trees | Robust to the skewed AQI distribution |
| `gradient_boosting` | boosted trees | scikit-learn boosting |
| `xgboost` | boosted trees | Usually the strongest on tabular data |
| `tf_mlp` | TensorFlow | Dense net, BatchNorm + dropout, Huber loss |
| `tf_lstm` | TensorFlow | Stacked LSTM over the lag window as a pseudo-sequence |

Selection is by mean RMSE across the three horizons on a **chronological**
hold-out (the last three weeks). A random split would leak the future through
the lag features and produce R² numbers that don't survive deployment.

The persistence baseline is trained and scored alongside everything else, and
the training pipeline flags any run where the winner fails to beat it. With lag
features this dense, persistence is genuinely hard to beat at +24 h — a model
that "only" matches it has learned nothing, and that needs to be visible rather
than hidden behind a good-looking R².

Metrics reported: **RMSE, MAE and R²**, overall and per horizon.

---

## Automation

| Workflow | Schedule | What it does |
| --- | --- | --- |
| `feature_pipeline.yml` | hourly, `0 * * * *` | Fetch → engineer → write to store → refresh forecast → commit |
| `training_pipeline.yml` | daily, `30 2 * * *` | Retrain all models → promote best → SHAP → EDA → commit |
| `ci.yml` | every push / PR | Ruff, pytest, plus a 90-day end-to-end smoke run and an API boot check |

Each scheduled run commits the refreshed Parquet, model bundle and JSON
artifacts back to the repository. That commit is what makes the deployment
genuinely live: Streamlit Community Cloud redeploys on push, so the dashboard
picks up new data and new models with no manual intervention.

The hourly job deliberately installs only pandas/numpy/requests — no ML
frameworks — so it finishes in well under a minute.

`airflow/dags/aqi_dag.py` provides the same two schedules as Airflow DAGs. The
tasks are thin `PythonOperator`s over the identical `run()` functions, so
switching orchestrators changes no pipeline code.

---

## Explainability

**Global** — mean |SHAP| over the hold-out set, computed during training and
cached to `data/artifacts/shap_summary.json` so the dashboard renders instantly.
Tree models use the exact `TreeExplainer`; Ridge and the Keras nets fall back to
`KernelExplainer` over a k-means background.

**Local** — per-prediction contributions on demand, via SHAP (`/explain?local=1`
or the dashboard button) or LIME (`src.explain.lime_explain`), which fits a
linear surrogate around the single point and is often easier to put in front of
a non-technical reader.

---

## Alerting

Fires when observed or forecast AQI reaches `AQI_ALERT_THRESHOLD` (default 150 —
the top of "Unhealthy for Sensitive Groups", one point below where "Unhealthy"
begins, so the alert errs towards warning). Forecast alerts are the useful ones:
a warning 72 hours ahead is actionable in a way that a warning about right now
is not.

Alerts surface in three places: the dashboard banner, the pipeline logs, and a
webhook (`AQI_ALERT_WEBHOOK`) that accepts Slack and Discord incoming-webhook
URLs. With no webhook configured, alerts are still returned and logged —
alerting never becomes a hard dependency of the pipeline.

---

## Monitoring

The Flask API exposes Prometheus metrics at `/metrics`:

| Metric | Meaning |
| --- | --- |
| `aqi_pipeline_runs_total` | Runs by pipeline and status |
| `aqi_pipeline_duration_seconds` | Wall-clock histogram |
| `aqi_pipeline_last_success_timestamp` | Freshness of each pipeline |
| `aqi_model_rmse` / `_mae` / `_r2` | Production model accuracy (drift detection) |
| `aqi_current`, `aqi_forecast` | Observed and forecast AQI |
| `aqi_feature_age_seconds` | Staleness of the newest feature row |
| `aqi_api_requests_total`, `aqi_api_latency_seconds` | API traffic and latency |

```bash
docker compose up -d
```

Grafana on <http://localhost:3000> (admin/admin) with the dashboard
pre-provisioned; Prometheus on <http://localhost:9090>. Alert rules in
`monitoring/alert_rules.yml` cover hazardous AQI, stale pipelines, stale data,
degraded model accuracy and API error rate.

Because GitHub Actions runners are ephemeral, in-process counters die with the
job. Metrics are therefore mirrored to `data/artifacts/monitoring_state.json`,
committed back, and re-read on boot so the gauges show continuity rather than
resetting to zero every hour.

---

## API reference

| Endpoint | Description |
| --- | --- |
| `GET /health` | Liveness plus feature freshness and model availability |
| `GET /predict` | 3-day forecast (`?live=1` bypasses the feature store) |
| `GET /current` | Latest observation, with an AQICN cross-check when a token is set |
| `GET /history?hours=` | Observed AQI and pollutant series |
| `GET /backtest?days=` | Predicted vs actual replay with per-horizon error |
| `GET /models` | Model comparison table and registry history |
| `GET /explain?horizon=&local=` | SHAP explanation, global or per-prediction |
| `GET /alerts` | Active hazardous-AQI alerts |
| `GET /metrics` | Prometheus exposition |

---

## Dashboard

| Tab | Contents |
| --- | --- |
| **Forecast** | Current AQI, three horizon cards, trajectory chart with an 80% band, current conditions, active alerts |
| **History** | Observed AQI and pollutants over a selectable window, plus predicted-vs-actual backtest |
| **EDA** | Daily/weekly cycles, distribution, autocorrelation, correlation matrix, full written report |
| **Explainability** | Global SHAP importance and an on-demand local explanation |
| **Models** | Comparison table, RMSE by model, error growth with horizon, registry history |
| **Pipeline** | Feature freshness, last run summaries, monitoring state, automation schedule |

---

## Testing

```bash
pytest tests -v
```

Coverage focuses on the parts where a silent error would be invisible
downstream: the EPA breakpoint table (every target derives from it), feature
causality (a leak inflates metrics rather than breaking anything), feature-store
upsert semantics, model round-tripping, and alert thresholds.

---

## Configuration

All settings are environment variables with working defaults — see
`.env.example`. The ones that change behaviour most:

| Variable | Default | Effect |
| --- | --- | --- |
| `AQI_CITY` / `AQI_LAT` / `AQI_LON` | Karachi | Target location |
| `AQI_DATA_SOURCE` | `auto` | `auto` picks OpenWeather if a key exists, else Open-Meteo |
| `AQI_FEATURE_STORE` | `auto` | `auto` picks Hopsworks if credentials exist, else local Parquet |
| `AQI_ALERT_THRESHOLD` | `150` | AQI at which alerts fire |
| `AQI_BACKFILL_DAYS` | `365` | Default backfill window |

Both `auto` resolvers degrade rather than fail: if Hopsworks is unreachable or
an OpenWeather key has expired, the run falls back to the local store or
Open-Meteo and logs a warning instead of taking the pipeline down.

---

## Deployment

**Dashboard** — push to GitHub, then point Streamlit Community Cloud at
`app/streamlit_app.py`. The scheduled workflows commit fresh data, and Streamlit
redeploys on push.

**API** — `docker compose up api`, or any container host. `gunicorn` (Linux) and
`waitress` (Windows) are both in `requirements.txt`.

**Secrets** — add `OPENWEATHER_API_KEY`, `AQICN_TOKEN`, `HOPSWORKS_API_KEY`,
`HOPSWORKS_PROJECT` and `AQI_ALERT_WEBHOOK` as GitHub Actions repository
secrets, and `AQI_CITY` / `AQI_LAT` / `AQI_LON` as repository variables.

---

## Known limits

- **Accuracy degrades with horizon.** AQI autocorrelation decays substantially
  past ~72 h, which is why the forecast stops there. The +72 h number is
  meaningfully less reliable than +24 h; the dashboard shows the widening
  uncertainty band rather than hiding it.
- **The uncertainty band is not a formal prediction interval.** It is
  ±1.28 × hold-out RMSE, which assumes roughly normal, homoscedastic residuals.
  Residuals here are neither — they fan out during pollution episodes. Treat the
  band as indicative.
- **Weather is treated as a current-state feature, not a forecast.** Feeding in
  forecast meteorology (which is itself available 72 h ahead) would likely be
  the single biggest accuracy improvement available.
- **One city per deployment.** The schema and pipelines are city-scoped;
  multi-city would need a location dimension in the feature group.
