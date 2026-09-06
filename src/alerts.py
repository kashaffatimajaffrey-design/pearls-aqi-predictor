"""Hazardous-AQI alerting.

Two entry points, both returning a plain list of alert dicts so the caller can
render them (dashboard), log them (pipelines) or ship them (webhook):

* evaluate_observation_alert -- fires on the current measured AQI.
* evaluate_forecast_alerts   -- fires on any horizon in the forecast, which is
                                the useful one: a warning 72 hours ahead is
                                actionable in a way that a warning about right
                                now is not.

Delivery is a generic webhook (Slack and Discord both accept a JSON body with a
`text`/`content` field), configured via AQI_ALERT_WEBHOOK. With no webhook set
the alert is still returned and logged -- alerting never becomes a hard
dependency of the pipeline.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import requests

from . import config
from .aqi import advice, categorize

log = logging.getLogger(__name__)

SEVERITY = {
    "Good": 0,
    "Moderate": 1,
    "Unhealthy for Sensitive Groups": 2,
    "Unhealthy": 3,
    "Very Unhealthy": 4,
    "Hazardous": 5,
}

DEFAULT_EMOJI = "\N{WARNING SIGN}"
EMOJI = {
    "Unhealthy for Sensitive Groups": "\N{LARGE ORANGE CIRCLE}",
    "Unhealthy": "\N{LARGE RED CIRCLE}",
    "Very Unhealthy": "\N{LARGE PURPLE CIRCLE}",
    "Hazardous": "\N{SKULL}",
}


def _alert(kind: str, aqi: float, when, *, horizon_hours: int | None = None) -> dict:
    category = categorize(aqi)
    icon = EMOJI.get(category, DEFAULT_EMOJI)
    return {
        "kind": kind,
        "aqi": round(float(aqi), 1),
        "category": category,
        "severity": SEVERITY.get(category, 0),
        "horizon_hours": horizon_hours,
        "when": str(when),
        "threshold": config.ALERT_AQI_THRESHOLD,
        "message": (
            f"{icon} {config.CITY}: AQI {round(float(aqi))} ({category})"
            + (f" forecast {horizon_hours}h ahead" if horizon_hours else " right now")
        ),
        "advice": advice(aqi),
        "raised_at": datetime.now(UTC).isoformat(),
    }


def _alert_key(alert: dict) -> str:
    """Identity of an alert for deduplication.

    Keyed on kind, horizon and severity band -- deliberately NOT on the exact
    AQI value, so a pollution episode drifting from 171 to 174 does not read as
    a new event.
    """
    return f"{alert['kind']}:{alert.get('horizon_hours') or 'now'}:{alert['category']}"


def _suppressed(alerts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split alerts into (to_send, suppressed) using the cooldown window.

    Without this, the hourly pipeline re-sends an identical webhook every run:
    a three-day episode becomes 72 identical Slack messages and people mute the
    channel, which is worse than no alerting at all. An alert re-fires only
    after ALERT_COOLDOWN_HOURS, or immediately if its severity has worsened.
    """
    state = _load_alert_state()
    now = datetime.now(UTC)
    cooldown = timedelta(hours=config.ALERT_COOLDOWN_HOURS)

    to_send, suppressed = [], []
    for alert in alerts:
        key = _alert_key(alert)
        previous = state.get(key)
        escalated = previous is not None and alert["severity"] > previous.get("severity", 0)

        if previous and not escalated:
            try:
                if now - datetime.fromisoformat(previous["sent_at"]) < cooldown:
                    suppressed.append(alert)
                    continue
            except (ValueError, KeyError):
                pass  # unparseable state -> treat as never sent

        to_send.append(alert)
        state[key] = {"sent_at": now.isoformat(), "severity": alert["severity"]}

    _save_alert_state(state)
    return to_send, suppressed


def _load_alert_state() -> dict:
    path = config.ARTIFACT_DIR / "alert_state.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save_alert_state(state: dict) -> None:
    (config.ARTIFACT_DIR / "alert_state.json").write_text(
        json.dumps(state, indent=2, default=str)
    )


def evaluate_observation_alert(aqi: float, ts) -> dict | None:
    """Alert on the latest measured AQI, or None if it is below threshold."""
    if aqi is None or aqi < config.ALERT_AQI_THRESHOLD:
        return None
    alert = _alert("observation", aqi, ts)
    dispatch([alert])
    return alert


def evaluate_forecast_alerts(forecasts: list[dict]) -> list[dict]:
    """Alert on any forecast horizon at or above the threshold.

    Returns every breaching horizon so the dashboard can display them all;
    `dispatch` decides which are actually worth sending to a webhook.
    """
    alerts = [
        _alert("forecast", f["aqi"], f["valid_at"], horizon_hours=f["horizon_hours"])
        for f in forecasts
        if f.get("aqi") is not None and f["aqi"] >= config.ALERT_AQI_THRESHOLD
    ]
    if alerts:
        dispatch(alerts)
    return alerts


def dispatch(alerts: list[dict]) -> bool:
    """Send alerts to the configured webhook, honouring the cooldown.

    Returns True if something was actually delivered.
    """
    for a in alerts:
        log.warning("AQI ALERT: %s", a["message"])
    append_history(alerts)

    alerts, held = _suppressed(alerts)
    if held:
        log.info("%d alert(s) suppressed by the %dh cooldown",
                 len(held), config.ALERT_COOLDOWN_HOURS)

    if not alerts or not config.ALERT_WEBHOOK_URL:
        return False

    lines = [f"*Air quality alert -- {config.CITY}*"]
    for a in alerts:
        lines.append(f"- {a['message']}")
        lines.append(f"  _{a['advice']}_")
    text = "\n".join(lines)

    try:
        r = requests.post(
            config.ALERT_WEBHOOK_URL,
            json={"text": text, "content": text},  # Slack uses text, Discord content
            timeout=15,
        )
        r.raise_for_status()
        log.info("dispatched %d alert(s) to webhook", len(alerts))
        return True
    except Exception as exc:  # pragma: no cover - network
        log.warning("alert dispatch failed: %s", exc)
        return False


def alert_history_path():
    return config.ARTIFACT_DIR / "alert_history.json"


def append_history(alerts: list[dict], keep: int = 200) -> None:
    """Persist alerts so the dashboard can show a recent-alerts panel."""
    if not alerts:
        return
    path = alert_history_path()
    history = []
    if path.exists():
        try:
            history = json.loads(path.read_text())
        except json.JSONDecodeError:
            history = []
    history.extend(alerts)
    path.write_text(json.dumps(history[-keep:], indent=2, default=str))
