"""Engagement layer: streaks, badges, activity scoring and a model report card.

Kept out of the Streamlit file deliberately -- this is logic with edge cases
(what counts as a "clean day"? what happens at a data gap?), so it belongs
somewhere testable rather than inline in the UI.

Design rule followed throughout: gamification decorates the *app*, never the
*health signal*. Nothing here changes an AQI number or its EPA colour. A streak
breaking is a nudge; the red banner above it is the actual warning.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------- theming ---
# Blue app chrome. AQI values keep their EPA colours -- see module docstring.
SKY = {
    # Every phase is LIGHT with dark ink. The sky still shifts through the day,
    # but readability is not allowed to depend on the time of day -- a dark page
    # meant fighting Streamlit's light-theme components on every widget, and
    # losing. Hue carries the mood; contrast stays constant.
    "dawn":  {"from": "#fbe6dd", "to": "#eef4fb", "ink": "#0f2740", "label": "Dawn"},
    "day":   {"from": "#cfe4fa", "to": "#f4f9ff", "ink": "#0f2740", "label": "Daytime"},
    "dusk":  {"from": "#e3dcf2", "to": "#eef4fb", "ink": "#0f2740", "label": "Dusk"},
    "night": {"from": "#c6d6ea", "to": "#e9f0f9", "ink": "#0f2740", "label": "Night"},
}

# Forecast depth: further out = deeper blue = less certain.
HORIZON_BLUE = ["#5b9bd5", "#2f6fb0", "#17456f"]


def sky_phase(when: datetime | None = None, tz: str | None = None) -> dict:
    """Which sky the app should wear, from local clock time."""
    tz = tz or config.TIMEZONE
    now = pd.Timestamp(when or pd.Timestamp.now(tz="UTC"))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    hour = now.tz_convert(tz).hour
    if 5 <= hour < 8:
        phase = "dawn"
    elif 8 <= hour < 17:
        phase = "day"
    elif 17 <= hour < 20:
        phase = "dusk"
    else:
        phase = "night"
    return {"phase": phase, **SKY[phase]}


def horizon_blue(index: int) -> str:
    return HORIZON_BLUE[min(index, len(HORIZON_BLUE) - 1)]


# ---------------------------------------------------------------- streaks ---
def daily_peaks(df: pd.DataFrame, tz: str | None = None) -> pd.Series:
    """Worst AQI per local day. Peak, not mean: one hazardous afternoon should
    not be averaged away by a clean night."""
    if df is None or df.empty or "aqi" not in df:
        return pd.Series(dtype=float)
    tz = tz or config.TIMEZONE
    local = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(tz)
    return df.assign(_day=local.dt.date).groupby("_day")["aqi"].max()


def clean_streak(df: pd.DataFrame, threshold: int | None = None,
                 tz: str | None = None) -> dict:
    """Consecutive clean days ending at the most recent complete day.

    A "clean day" is one whose *peak* AQI stayed under the threshold. The
    current day is excluded because it is still in progress and would flicker.
    """
    threshold = threshold or config.ALERT_AQI_THRESHOLD
    peaks = daily_peaks(df, tz)
    if peaks.empty:
        return {"current": 0, "best": 0, "threshold": threshold, "days_tracked": 0}

    peaks = peaks.iloc[:-1] if len(peaks) > 1 else peaks  # drop the partial day
    clean = (peaks < threshold).to_numpy()

    current = 0
    for ok in clean[::-1]:
        if not ok:
            break
        current += 1

    best = longest = 0
    for ok in clean:
        longest = longest + 1 if ok else 0
        best = max(best, longest)

    return {
        "current": int(current),
        "best": int(best),
        "threshold": int(threshold),
        "days_tracked": int(len(peaks)),
        "clean_pct": round(100 * float(clean.mean()), 1) if len(clean) else 0.0,
    }


# ----------------------------------------------------------------- levels ---
LEVELS = [
    (0, "Weather Watcher"), (7, "Air Apprentice"), (30, "Sky Reader"),
    (90, "Atmosphere Analyst"), (180, "Air Quality Adept"), (365, "Climate Sentinel"),
]


def level_for(days_tracked: int) -> dict:
    """Level from days of data held. Progress toward the next tier included so
    the UI can draw a bar rather than a bare number."""
    idx = 0
    for i, (need, _) in enumerate(LEVELS):
        if days_tracked >= need:
            idx = i
    need, title = LEVELS[idx]
    if idx + 1 < len(LEVELS):
        nxt_need, nxt_title = LEVELS[idx + 1]
        span = max(1, nxt_need - need)
        progress = min(1.0, (days_tracked - need) / span)
    else:
        nxt_need, nxt_title, progress = None, None, 1.0
    return {
        "level": idx + 1, "title": title, "days_tracked": int(days_tracked),
        "next_title": nxt_title, "next_at": nxt_need, "progress": round(progress, 3),
    }


# ----------------------------------------------------------------- badges ---
def badges(df: pd.DataFrame, streak: dict, stats: dict | None = None) -> list[dict]:
    """Achievements earned from the actual record. Every one is derived from
    data, never granted for merely opening the page."""
    stats = stats or {}
    peaks = daily_peaks(df)
    days = len(peaks)
    out: list[dict] = []

    def add(key, icon, title, desc, earned, progress=None):
        out.append({"key": key, "icon": icon, "title": title, "description": desc,
                    "earned": bool(earned), "progress": progress})

    add("first_week", "\N{SEEDLING}", "First Week",
        "Seven days of air quality tracked", days >= 7, min(1.0, days / 7))
    add("full_season", "\N{MAPLE LEAF}", "Full Season",
        "Ninety days of history collected", days >= 90, min(1.0, days / 90))
    add("full_year", "\N{EARTH GLOBE ASIA-AUSTRALIA}", "Year of Air",
        "A complete year in the feature store", days >= 365, min(1.0, days / 365))
    add("streak_7", "\N{FIRE}", "Clean Week",
        "Seven consecutive days below the alert threshold",
        streak.get("best", 0) >= 7, min(1.0, streak.get("best", 0) / 7))
    add("streak_30", "\N{GLOWING STAR}", "Clean Month",
        "Thirty consecutive clean days",
        streak.get("best", 0) >= 30, min(1.0, streak.get("best", 0) / 30))

    if not peaks.empty:
        add("survived_spike", "\N{SHIELD}", "Weathered the Spike",
            "Recorded an episode above AQI 150", bool((peaks >= 150).any()))
        add("blue_sky", "\N{SUN WITH FACE}", "Blue Sky Day",
            "Logged a day that never left the Good band", bool((peaks <= 50).any()))
    if stats.get("aqi_max"):
        add("extremes", "\N{HIGH VOLTAGE SIGN}", "Full Spectrum",
            "Observed both Good and Unhealthy conditions",
            stats.get("aqi_min", 999) <= 50 and stats.get("aqi_max", 0) >= 150)
    return out


# -------------------------------------------------------- activity scoring --
ACTIVITIES = [
    ("Outdoor run / cycling", 60),
    ("Walk or light exercise", 100),
    ("Children playing outside", 80),
    ("Windows open for airflow", 100),
    ("Strenuous sport", 55),
]


def outdoor_score(aqi: float) -> dict:
    """A 0-100 'how good is it out there' score plus per-activity guidance.

    The score is inverted AQI on a 0-200 scale, which keeps it monotonic and
    easy to read. Activity thresholds are the AQI above which each becomes
    inadvisable.
    """
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return {"score": None, "verdict": "Unknown", "activities": []}

    score = int(round(max(0.0, min(100.0, 100 * (1 - min(aqi, 200) / 200)))))
    if aqi <= 50:
        verdict = "Perfect day to be outside"
    elif aqi <= 100:
        verdict = "Fine for most people"
    elif aqi <= 150:
        verdict = "Sensitive groups should take care"
    elif aqi <= 200:
        verdict = "Limit time outdoors"
    else:
        verdict = "Stay inside"

    return {
        "score": score,
        "verdict": verdict,
        "activities": [
            {"name": name, "ok": aqi <= limit, "limit": limit}
            for name, limit in ACTIVITIES
        ],
    }


# ------------------------------------------------------------ report card ---
GRADES = [(3.0, "A+"), (5.0, "A"), (7.5, "B"), (10.0, "C"), (15.0, "D")]


def model_report_card(backtest: pd.DataFrame) -> dict | None:
    """Grade the model on its own recent track record.

    Uses the backtest rather than the training metrics, so this reflects how the
    production model actually performed against observed AQI -- an honest
    scoreboard, not a marketing number.
    """
    if backtest is None or backtest.empty:
        return None

    per_horizon = []
    for h, g in backtest.groupby("horizon_hours"):
        mae = float(g["abs_error"].mean())
        within10 = float((g["abs_error"] <= 10).mean() * 100)
        per_horizon.append({
            "horizon_hours": int(h), "mae": round(mae, 2),
            "rmse": round(float((g["error"] ** 2).mean() ** 0.5), 2),
            "hit_rate_10": round(within10, 1),
            "grade": _grade(mae), "n": int(len(g)),
        })

    overall_mae = float(backtest["abs_error"].mean())
    return {
        "overall_mae": round(overall_mae, 2),
        "overall_grade": _grade(overall_mae),
        "hit_rate_10": round(float((backtest["abs_error"] <= 10).mean() * 100), 1),
        "predictions_scored": int(len(backtest)),
        "per_horizon": per_horizon,
    }


def _grade(mae: float) -> str:
    for limit, grade in GRADES:
        if mae <= limit:
            return grade
    return "F"


# ----------------------------------------------------------- data freshness -
def freshness(df: pd.DataFrame) -> dict:
    """How live the data actually is -- surfaced prominently because a stale
    dashboard that looks fresh is worse than one that admits it."""
    if df is None or df.empty:
        return {"hours": None, "state": "empty", "label": "No data"}
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(df["ts"].max())).total_seconds() / 3600
    if age <= 2:
        state, label = "live", "Live"
    elif age <= 6:
        state, label = "recent", "Recent"
    else:
        state, label = "stale", "Stale"
    return {"hours": round(age, 1), "state": state, "label": label}


def next_refresh(df: pd.DataFrame) -> str:
    """When the hourly pipeline should next land, for a countdown."""
    if df is None or df.empty:
        return "unknown"
    latest = pd.Timestamp(df["ts"].max())
    nxt = (latest + timedelta(hours=1)).tz_convert(config.TIMEZONE)
    return nxt.strftime("%H:%M")
