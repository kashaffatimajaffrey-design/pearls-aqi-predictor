# Requirements traceability

Every requirement from the brief, mapped to where it lives and — more usefully —
**what level of evidence stands behind it**. The distinction that matters is
between code that has been *run and observed*, code that is *unit-tested but not
exercised in production*, and code that is *written but never executed* because
something external is missing.

Legend:

| | Meaning |
| --- | --- |
| **Verified** | Executed and the output observed in this build |
| **Partial** | Runs, but one path or schedule has never been exercised |
| **Unexercised** | Implemented; blocked on a credential, a remote, or a daemon |

---

## Technology stack

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Python | **Verified** | 3.13 local, 3.11/3.12 in CI matrix |
| 2 | Scikit-learn | **Verified** | Ridge, RandomForest, GradientBoosting, Pipeline, imputer, metrics — all trained |
| 3 | TensorFlow | **Verified** | MLP, LSTM and a real encoder-decoder sequence model all trained (RMSE 20.00 / 11.76); all lost to trees, results kept |
| 4 | XGBoost | **Verified** | Trained and compared; `gradient_boosting` currently wins at RMSE 7.74, R² 0.361 |
| 5 | Hopsworks *or* Vertex AI | **Unexercised** | `src/store/` adapters written; no account. Fallback to local store verified degrading cleanly |
| 6 | Airflow *or* GitHub Actions | **Unexercised** | Both written. Actions never fired (no remote); DAG never parsed (`apache-airflow` not installed) |
| 7 | Streamlit | **Verified** | All seven tabs rendered in a browser; zero exceptions, zero deprecation warnings in the server log |
| 8 | Flask | **Verified** | All nine endpoints exercised with curl |
| 9 | AQICN *or* OpenWeather | **Unexercised** | Both connectors written; no keys. Live data came from Open-Meteo |
| 10 | SHAP | **Verified** | TreeExplainer over 300 hold-out rows; rankings in `data/artifacts/shap_summary.json` |
| 11 | Git | **Verified** | 9 commits with full history |
| 12 | Prometheus + Grafana *(self-imposed)* | **Unexercised** | `docker compose config` validates; Docker daemon not running, so never scraped |

## Scope

| # | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| 13 | Feature pipeline, hourly | **Partial** | Pipeline runs and writes correctly; the hourly cron has never fired |
| 14 | Historical backfill | **Verified** | 8,670 rows, 365 days, 99% hourly coverage |
| 15 | Training pipeline, daily | **Partial** | Runs end to end; the daily cron has never fired |
| 16 | CI/CD automation | **Unexercised** | Three workflows written; no remote, so none has ever run |
| 17 | Dashboard, real-time 3-day | **Verified** | Rendered live with sky theme, gamified Progress tab, uncertainty bands |
| 18 | EDA | **Verified** | `reports/eda_report.md` + 7 figures + notebook |
| 19 | SHAP **or** LIME | **Verified** | Both. LIME executed: local R² 0.241 |
| 20 | Hazardous-AQI alerts | **Partial** | Logic + cooldown covered by tests; never fired live — Karachi AQI has not crossed 150 in the window |
| 21 | Multiple models compared | **Verified** | Seven families plus two sequence models, table in `data/artifacts/model_comparison.json` |
| 22 | **Genuinely live, no manual entry** | **Unproven** | Architecturally complete; unproven until pushed and the cron fires |

---

## Where a substitution was made, and why

### Open-Meteo instead of AQICN / OpenWeather

The brief names AQICN or OpenWeather. Both connectors are implemented and
selected automatically when a key is present, but every number in this repo came
from Open-Meteo, which needs no key.

That was a deliberate call: it makes the project reproducible by anyone who
clones it, and Open-Meteo offers gapless hourly history that the free AQICN tier
does not. The honest cost is that Open-Meteo serves **CAMS model output rather
than ground-station measurement**, so we are forecasting a physical model's
estimate of air quality. See the README's provenance note.

**This is a deviation from the brief and should be declared, not glossed.**
Adding an OpenWeather key (free, ~5 minutes) makes the project satisfy the
requirement literally, with no code change.

### Local Parquet store instead of Hopsworks

`LocalFeatureStore` and `LocalModelRegistry` implement the same interface as the
Hopsworks adapters — upsert on primary key, event-time dedupe, versioned
bundles, a production pointer, and history. They are what actually held the
8,670 feature rows and every registered model version.

This is a working substitute rather than a stub, and swapping backends is one
environment variable. But it is a substitute: no Hopsworks project has been
created, so that adapter is unproven.

### GitHub Actions as primary, Airflow as alternative

The brief says "Airflow **or** GitHub Actions", so this is satisfied by the
disjunction. Actions is primary because it is genuinely serverless, which is the
premise of the project. The Airflow DAGs exist so the same pipelines can run on a
self-hosted scheduler, and they call the identical `run()` functions — but
`apache-airflow` is not installed here, so they have never been parsed.

---

## What closes the remaining gaps

Ordered by value, with rough effort:

| Action | Closes | Effort |
| --- | --- | --- |
| **Push to GitHub and let the cron fire** | #16, #22, and upgrades #13/#15 to Verified | 10 min + a day of waiting |
| Free OpenWeather API key | #9 literally | 5 min, no code change |
| Free Hopsworks account | #5 literally | 20 min, no code change |
| Start Docker Desktop, `docker compose up` | #12 | 10 min |
| Wait for an AQI episode, or lower the threshold to observe | #20 firing live | passive |

Requirement #22 is the one that matters most. Everything needed for it is
written; it simply has not been *demonstrated*, and until the workflow history
exists the claim should be stated as "architecturally complete" rather than
"running".
