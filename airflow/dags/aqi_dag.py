"""Airflow DAGs -- the self-hosted alternative to the GitHub Actions workflows.

GitHub Actions is the primary orchestrator (it is genuinely serverless, which is
the point of the project). These DAGs exist so the same pipelines can be run on
an Airflow deployment without changing any pipeline code: each task is a thin
PythonOperator over the same `run()` functions the workflows call.

Drop this file into $AIRFLOW_HOME/dags/ with the project importable on PYTHONPATH.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow.operators.python import PythonOperator  # noqa: E402

from airflow import DAG  # noqa: E402

DEFAULT_ARGS = {
    "owner": "aqi-predictor",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
    "email_on_failure": False,
}


def _feature(**_):
    from src.pipelines.feature_pipeline import run

    return run(hours=168)


def _train(**_):
    from src.pipelines.training_pipeline import run

    return run()


def _eda(**_):
    from src.eda import run

    return run()["stats"]


def _forecast(**_):
    from src.pipelines.inference import clear_cache, predict

    clear_cache()  # the daily DAG has just written a new model
    return predict()


with DAG(
    dag_id="aqi_feature_pipeline",
    description="Hourly AQI feature ingestion into the Feature Store",
    default_args=DEFAULT_ARGS,
    schedule="0 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aqi", "features", "hourly"],
) as feature_dag:
    ingest = PythonOperator(task_id="fetch_and_engineer_features", python_callable=_feature)
    forecast = PythonOperator(task_id="refresh_forecast", python_callable=_forecast)

    ingest >> forecast


with DAG(
    dag_id="aqi_training_pipeline",
    description="Daily model retraining, evaluation and registration",
    default_args=DEFAULT_ARGS,
    schedule="30 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aqi", "training", "daily"],
) as training_dag:
    train = PythonOperator(task_id="train_and_register", python_callable=_train)
    eda = PythonOperator(task_id="refresh_eda_report", python_callable=_eda)
    predict = PythonOperator(task_id="refresh_forecast", python_callable=_forecast)

    train >> [eda, predict]
