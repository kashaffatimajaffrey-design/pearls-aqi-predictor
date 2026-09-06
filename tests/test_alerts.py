"""Tests for the hazardous-AQI alerting logic."""
from __future__ import annotations

import pandas as pd

from src import config
from src.alerts import SEVERITY, evaluate_forecast_alerts, evaluate_observation_alert


def _forecast(aqi, horizon=24):
    return {"aqi": aqi, "horizon_hours": horizon, "valid_at": "2024-01-02T00:00:00+00:00"}


class TestObservationAlerts:
    def test_silent_below_threshold(self):
        assert evaluate_observation_alert(80, pd.Timestamp.utcnow()) is None

    def test_fires_at_the_threshold(self):
        # 150 is the *top* of "Unhealthy for Sensitive Groups" -- "Unhealthy"
        # starts at 151 -- so the default threshold fires one point early. That
        # is deliberate: an air-quality alert should err towards warning.
        alert = evaluate_observation_alert(config.ALERT_AQI_THRESHOLD, pd.Timestamp.utcnow())
        assert alert is not None
        assert alert["kind"] == "observation"
        assert alert["category"] == "Unhealthy for Sensitive Groups"

    def test_fires_for_unhealthy_air(self):
        alert = evaluate_observation_alert(175, pd.Timestamp.utcnow())
        assert alert is not None
        assert alert["category"] == "Unhealthy"

    def test_carries_advice_and_severity(self):
        alert = evaluate_observation_alert(320, pd.Timestamp.utcnow())
        assert alert["category"] == "Hazardous"
        assert alert["severity"] == SEVERITY["Hazardous"]
        assert len(alert["advice"]) > 10

    def test_none_input_is_safe(self):
        assert evaluate_observation_alert(None, pd.Timestamp.utcnow()) is None


class TestForecastAlerts:
    def test_no_alerts_when_all_clean(self):
        assert evaluate_forecast_alerts([_forecast(40), _forecast(60, 48)]) == []

    def test_one_alert_per_breaching_horizon(self):
        alerts = evaluate_forecast_alerts([
            _forecast(40, 24), _forecast(180, 48), _forecast(260, 72),
        ])
        assert len(alerts) == 2
        assert {a["horizon_hours"] for a in alerts} == {48, 72}

    def test_alert_names_its_horizon_in_the_message(self):
        alerts = evaluate_forecast_alerts([_forecast(200, 72)])
        assert "72h ahead" in alerts[0]["message"]
        assert alerts[0]["kind"] == "forecast"

    def test_severity_orders_categories(self):
        alerts = evaluate_forecast_alerts([_forecast(160, 24), _forecast(310, 48)])
        by_horizon = {a["horizon_hours"]: a["severity"] for a in alerts}
        assert by_horizon[48] > by_horizon[24]

    def test_empty_forecast_is_safe(self):
        assert evaluate_forecast_alerts([]) == []
