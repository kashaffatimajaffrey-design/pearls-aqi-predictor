"""Tests for alert deduplication.

The bug these guard against: the hourly pipeline re-sending an identical
webhook every run, so a three-day pollution episode becomes 72 identical
messages and people mute the channel.
"""
from __future__ import annotations

import json

import pytest

from src import alerts, config


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point alert state and history at a temp dir, not the real artifacts."""
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", "")  # never hit the network
    return tmp_path


def forecast(aqi, horizon=24):
    return {"aqi": aqi, "horizon_hours": horizon, "valid_at": "2024-01-02T00:00:00+00:00"}


class TestCooldown:
    def test_first_alert_is_sent(self):
        to_send, held = alerts._suppressed([alerts._alert("forecast", 180, "x", horizon_hours=24)])
        assert len(to_send) == 1 and not held

    def test_identical_repeat_is_suppressed(self):
        a = alerts._alert("forecast", 180, "x", horizon_hours=24)
        alerts._suppressed([a])
        to_send, held = alerts._suppressed([a])
        assert not to_send and len(held) == 1

    def test_small_drift_within_a_band_is_suppressed(self):
        """171 -> 174 is the same event, not a new one."""
        alerts._suppressed([alerts._alert("forecast", 171, "x", horizon_hours=24)])
        to_send, held = alerts._suppressed(
            [alerts._alert("forecast", 174, "x", horizon_hours=24)]
        )
        assert not to_send and len(held) == 1

    def test_escalation_breaks_through_immediately(self):
        """Getting worse must always notify, cooldown or not."""
        alerts._suppressed([alerts._alert("forecast", 160, "x", horizon_hours=24)])  # Unhealthy
        to_send, held = alerts._suppressed(
            [alerts._alert("forecast", 320, "x", horizon_hours=24)]  # Hazardous
        )
        assert len(to_send) == 1 and not held

    def test_different_horizons_alert_independently(self):
        alerts._suppressed([alerts._alert("forecast", 180, "x", horizon_hours=24)])
        to_send, _ = alerts._suppressed([alerts._alert("forecast", 180, "x", horizon_hours=72)])
        assert len(to_send) == 1

    def test_expired_cooldown_allows_resend(self, monkeypatch):
        a = alerts._alert("forecast", 180, "x", horizon_hours=24)
        alerts._suppressed([a])
        monkeypatch.setattr(config, "ALERT_COOLDOWN_HOURS", 0)
        to_send, held = alerts._suppressed([a])
        assert len(to_send) == 1 and not held

    def test_corrupt_state_file_does_not_crash(self, isolated_state):
        (isolated_state / "alert_state.json").write_text("{not json")
        to_send, _ = alerts._suppressed([alerts._alert("forecast", 180, "x", horizon_hours=24)])
        assert len(to_send) == 1


class TestHistory:
    def test_dispatch_records_history(self, isolated_state):
        alerts.evaluate_forecast_alerts([forecast(180), forecast(200, 48)])
        history = json.loads((isolated_state / "alert_history.json").read_text())
        assert len(history) == 2

    def test_history_accumulates_across_runs(self, isolated_state):
        alerts.evaluate_forecast_alerts([forecast(180)])
        alerts.evaluate_forecast_alerts([forecast(190, 48)])
        history = json.loads((isolated_state / "alert_history.json").read_text())
        assert len(history) == 2

    def test_history_is_capped(self, isolated_state):
        for _ in range(30):
            alerts.append_history([alerts._alert("forecast", 180, "x")], keep=10)
        history = json.loads((isolated_state / "alert_history.json").read_text())
        assert len(history) == 10

    def test_dashboard_still_sees_suppressed_alerts(self, isolated_state):
        """Suppression is a delivery concern -- the UI must still show the risk."""
        alerts.evaluate_forecast_alerts([forecast(180)])
        active = alerts.evaluate_forecast_alerts([forecast(180)])
        assert len(active) == 1  # returned to the caller even though not re-sent
