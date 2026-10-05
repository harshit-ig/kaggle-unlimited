"""Quota math: Kaggle's 12h session cap and 30h/week, resetting Saturday 00:00 UTC."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kaggle_rotate.quota import (
    Budget,
    Ledger,
    iso,
    overlap_seconds,
    utcnow,
    week_end,
    week_start,
)


def test_week_start_is_the_most_recent_saturday_midnight_utc():
    # 2026-10-07 is a Wednesday.
    wednesday = datetime(2026, 10, 7, 13, 30, tzinfo=UTC)
    assert week_start(wednesday) == datetime(2026, 10, 3, 0, 0, tzinfo=UTC)
    # Exactly on the boundary must not jump forward a week.
    saturday = datetime(2026, 10, 3, 0, 0, tzinfo=UTC)
    assert week_start(saturday) == saturday
    assert week_end(saturday) == saturday + timedelta(days=7)


def test_week_start_never_runs_ahead_of_now():
    now = utcnow()
    assert week_start(now) <= now
    assert (now - week_start(now)) < timedelta(days=7)


def test_overlap_only_counts_the_intersection():
    a = datetime(2026, 10, 1, tzinfo=UTC)
    b = datetime(2026, 10, 5, tzinfo=UTC)
    assert (
        overlap_seconds(a, b, datetime(2026, 10, 3, tzinfo=UTC), datetime(2026, 10, 10, tzinfo=UTC))
        == 2 * 86400
    )
    assert (
        overlap_seconds(a, b, datetime(2026, 11, 1, tzinfo=UTC), datetime(2026, 11, 5, tzinfo=UTC))
        == 0
    )


def test_usage_is_clipped_to_the_current_week(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    start = week_start() - timedelta(days=3)
    ledger.open_session("a", "a/kernel", now=start)
    ledger.close_session(ledger.sessions()[-1], "done", now=week_start() + timedelta(hours=2))

    # Only the 2 hours inside this week count.
    assert ledger.used_hours("a") == 2.0


def test_live_sessions_accrue_until_closed(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    session = ledger.open_session("a", "a/kernel", now=utcnow() - timedelta(hours=3))
    assert session.is_live()
    assert 2.9 < session.elapsed_hours() < 3.1
    assert ledger.used_hours("a") > 2.9

    ledger.close_session(session, "rotated")
    assert not session.is_live()
    assert ledger.find_live("a") is None
    assert ledger.used_hours("a") > 2.9


def test_usage_is_tracked_per_account(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    now = utcnow()
    for account in ("a", "b"):
        session = ledger.open_session(account, f"{account}/k", now=now - timedelta(hours=4))
        ledger.close_session(session, "done", now=now)
    assert ledger.used_hours("a") == 4.0
    assert ledger.used_hours("b") == 4.0
    assert ledger.total_used_hours() == 8.0


def test_budget_reports_session_remaining(tmp_path):
    budget = Budget(session_limit_hours=12.0, weekly_limit_hours=30.0)
    now = utcnow()
    assert budget.session_remaining_hours(None, now) == 12.0

    session = SessionFixture(now - timedelta(hours=11, minutes=30))
    assert 0.4 < budget.session_remaining_hours(session, now) < 0.6


def test_can_start_respects_weekly_budget_and_reserve(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    budget = Budget(session_limit_hours=12.0, weekly_limit_hours=30.0)

    ok, reason = budget.can_start(ledger, "a", reserve_hours=1.5)
    assert ok, reason

    spent = ledger.open_session("a", "a/k", now=utcnow() - timedelta(hours=29))
    ledger.close_session(spent, "done")
    ok, reason = budget.can_start(ledger, "a", reserve_hours=1.5)
    assert not ok
    assert "weekly quota left" in reason

    # A reserve of zero (we are desperate and idle) still respects the cap.
    ok, reason = budget.can_start(ledger, "a", reserve_hours=0.0)
    assert ok, reason


def test_can_start_refuses_an_account_that_already_has_a_live_session(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.open_session("a", "a/k")
    budget = Budget(session_limit_hours=12.0, weekly_limit_hours=30.0)
    ok, reason = budget.can_start(ledger, "a")
    assert not ok
    assert "live session" in reason


def test_per_account_weekly_override(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json")
    budget = Budget(session_limit_hours=12.0, weekly_limit_hours=30.0)
    session = ledger.open_session("a", "a/k", now=utcnow() - timedelta(hours=20))
    ledger.close_session(session, "done")

    assert budget.weekly_remaining_hours(ledger, "a") == 10.0
    assert budget.weekly_remaining_hours(ledger, "a", account_limit=25.0) == 5.0


def test_ledger_round_trips_through_disk(tmp_path):
    path = tmp_path / "nested" / "ledger.json"
    ledger = Ledger(path)
    session = ledger.open_session("a", "a/kernel")
    ledger.close_session(session, "rotated")

    reloaded = Ledger(path)
    assert len(reloaded.sessions()) == 1
    assert reloaded.sessions()[0].stop_reason == "rotated"


def test_iso_is_utc():
    assert iso(datetime(2026, 10, 7, tzinfo=UTC)).endswith("+00:00")


class SessionFixture:
    """Minimal stand-in for a Session when only elapsed time matters."""

    def __init__(self, started: datetime) -> None:
        self._started = started

    def is_live(self, now=None) -> bool:
        return True

    def elapsed_hours(self, now=None) -> float:
        return ((now or utcnow()) - self._started).total_seconds() / 3600
