"""Tests for the engagement layer.

The edge cases here are the ones that would quietly produce a wrong number on
the dashboard: a partial current day inflating a streak, a data gap breaking it,
or an empty frame crashing the page on first run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src import gamification as game


def frame(daily_peaks_list, start="2024-01-01"):
    """Build hourly data whose per-LOCAL-day peak matches the given list.

    Days are anchored to local midnight, not UTC midnight: `daily_peaks` groups
    by local date, so a UTC-anchored fixture would straddle two local days per
    block and silently change the day count.
    """
    rows = []
    day = pd.Timestamp(start, tz=config.TIMEZONE)
    for peak in daily_peaks_list:
        for h in range(24):
            rows.append({"ts": (day + pd.Timedelta(hours=h)).tz_convert("UTC"),
                         "aqi": float(peak if h == 12 else peak * 0.5)})
        day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


class TestStreak:
    def test_counts_trailing_clean_days(self):
        # last day is dropped as "in progress", so add a sentinel
        s = game.clean_streak(frame([200, 50, 50, 50, 60]))
        assert s["current"] == 3

    def test_breaks_on_a_dirty_day(self):
        s = game.clean_streak(frame([50, 50, 300, 50, 60]))
        assert s["current"] == 1

    def test_best_tracks_the_longest_run_not_the_last(self):
        s = game.clean_streak(frame([50, 50, 50, 50, 400, 50, 60]))
        assert s["best"] == 4
        assert s["current"] == 1

    def test_current_day_is_excluded(self):
        """The in-progress day must not count, or the streak flickers all day."""
        s = game.clean_streak(frame([50, 50, 50]))
        assert s["days_tracked"] == 2

    def test_uses_daily_peak_not_mean(self):
        """One hazardous afternoon must break the streak even if the day
        averages clean."""
        s = game.clean_streak(frame([50, 400, 50]))
        assert s["current"] == 0

    def test_clean_pct(self):
        s = game.clean_streak(frame([50, 50, 400, 50, 60]))
        assert s["clean_pct"] == pytest.approx(75.0)

    def test_empty_frame_is_safe(self):
        s = game.clean_streak(pd.DataFrame())
        assert s == {"current": 0, "best": 0, "threshold": config.ALERT_AQI_THRESHOLD,
                     "days_tracked": 0}


class TestLevels:
    @pytest.mark.parametrize("days, level", [(0, 1), (7, 2), (30, 3), (90, 4),
                                             (180, 5), (365, 6), (10_000, 6)])
    def test_thresholds(self, days, level):
        assert game.level_for(days)["level"] == level

    def test_progress_is_bounded(self):
        for days in (0, 3, 45, 200, 5000):
            assert 0.0 <= game.level_for(days)["progress"] <= 1.0

    def test_max_level_has_no_next(self):
        top = game.level_for(10_000)
        assert top["next_title"] is None
        assert top["progress"] == 1.0


class TestOutdoorScore:
    def test_clean_air_scores_high(self):
        assert game.outdoor_score(10)["score"] > 90

    def test_hazardous_air_scores_zero(self):
        assert game.outdoor_score(400)["score"] == 0

    def test_score_is_monotonic(self):
        scores = [game.outdoor_score(a)["score"] for a in range(0, 250, 10)]
        assert scores == sorted(scores, reverse=True)

    def test_activities_gate_on_aqi(self):
        clean = {a["name"]: a["ok"] for a in game.outdoor_score(20)["activities"]}
        dirty = {a["name"]: a["ok"] for a in game.outdoor_score(180)["activities"]}
        assert all(clean.values())
        assert not any(dirty.values())

    def test_nan_is_handled(self):
        assert game.outdoor_score(float("nan"))["score"] is None


class TestBadges:
    def test_badges_are_earned_from_data_not_granted(self):
        df = frame([50] * 3)
        earned = {b["key"]: b["earned"] for b in game.badges(df, game.clean_streak(df))}
        assert not earned["first_week"]      # only 3 days
        assert not earned["full_year"]

    def test_long_history_earns_time_badges(self):
        df = frame([50] * 100)
        earned = {b["key"]: b["earned"] for b in game.badges(df, game.clean_streak(df))}
        assert earned["first_week"] and earned["full_season"]
        assert not earned["full_year"]

    def test_spike_badge_requires_a_real_spike(self):
        calm = frame([50] * 10)
        spiky = frame([50] * 8 + [200, 60])
        assert not _earned(calm, "survived_spike")
        assert _earned(spiky, "survived_spike")

    def test_progress_present_on_unearned_badges(self):
        df = frame([50] * 3)
        for b in game.badges(df, game.clean_streak(df)):
            if not b["earned"] and b["progress"] is not None:
                assert 0.0 <= b["progress"] <= 1.0


def _earned(df, key):
    return next(b["earned"] for b in game.badges(df, game.clean_streak(df)) if b["key"] == key)


class TestReportCard:
    @staticmethod
    def _backtest(errors, horizon=24):
        return pd.DataFrame({
            "horizon_hours": [horizon] * len(errors),
            "error": errors,
            "abs_error": np.abs(errors),
        })

    def test_accurate_model_gets_a_top_grade(self):
        card = game.model_report_card(self._backtest(np.full(50, 1.0)))
        assert card["overall_grade"] == "A+"

    def test_poor_model_fails(self):
        card = game.model_report_card(self._backtest(np.full(50, 40.0)))
        assert card["overall_grade"] == "F"

    def test_hit_rate_counts_within_10(self):
        errors = np.array([1.0] * 30 + [50.0] * 10)
        card = game.model_report_card(self._backtest(errors))
        assert card["hit_rate_10"] == pytest.approx(75.0)

    def test_empty_backtest_returns_none(self):
        assert game.model_report_card(pd.DataFrame()) is None


class TestSkyAndFreshness:
    @pytest.mark.parametrize("hour, phase", [(6, "dawn"), (12, "day"),
                                             (18, "dusk"), (23, "night"), (3, "night")])
    def test_sky_phase_by_local_hour(self, hour, phase):
        when = pd.Timestamp(f"2024-06-01 {hour:02d}:00", tz=config.TIMEZONE)
        assert game.sky_phase(when.tz_convert("UTC"))["phase"] == phase

    def test_every_phase_has_a_full_palette(self):
        for phase in game.SKY.values():
            assert {"from", "to", "ink", "label"} <= set(phase)

    def test_horizon_blue_darkens_and_clamps(self):
        assert game.horizon_blue(0) != game.horizon_blue(2)
        assert game.horizon_blue(99) == game.horizon_blue(2)

    def test_freshness_flags_stale_data(self):
        old = pd.DataFrame({"ts": [pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=12)],
                            "aqi": [50.0]})
        assert game.freshness(old)["state"] == "stale"

    def test_freshness_flags_live_data(self):
        now = pd.DataFrame({"ts": [pd.Timestamp.now(tz="UTC")], "aqi": [50.0]})
        assert game.freshness(now)["state"] == "live"

    def test_freshness_on_empty(self):
        assert game.freshness(pd.DataFrame())["state"] == "empty"
