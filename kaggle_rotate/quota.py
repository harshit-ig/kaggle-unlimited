"""Quota accounting for Kaggle's per-session and per-week GPU limits.

Kaggle caps a single notebook session (12h) and an account's weekly GPU time
(30h), and the weekly counter resets Saturday 00:00 UTC. The ledger records every
session we start so both budgets can be enforced locally before we burn GPU time.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

WEEKLY_RESET_WEEKDAY = 5  # Saturday, matching Kaggle's weekly quota reset


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def week_start(now: datetime | None = None) -> datetime:
    """Start of the current quota week: the most recent Saturday 00:00 UTC."""
    now = (now or utcnow()).astimezone(UTC)
    days_since = (now.weekday() - WEEKLY_RESET_WEEKDAY) % 7
    candidate = (now - timedelta(days=days_since)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if candidate > now:
        candidate -= timedelta(days=7)
    return candidate


def week_end(start: datetime | None = None) -> datetime:
    return (start or week_start()) + timedelta(days=7)


def overlap_seconds(
    a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime
) -> float:
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0.0, (hi - lo).total_seconds())


def hours(seconds: float) -> float:
    # Six decimals keeps sub-second sessions visible instead of rounding them to zero.
    return round(seconds / 3600.0, 6)


@dataclass
class Session:
    account: str
    kernel_ref: str
    started_at: str
    ended_at: str = ""
    stop_reason: str = ""
    # GPU seconds actually billed are only known after the session ends; we
    # approximate with wall time so budgeting stays conservative.
    billed_hours: float | None = None

    @property
    def start(self) -> datetime:
        return datetime.fromisoformat(self.started_at)

    @property
    def end(self) -> datetime | None:
        return datetime.fromisoformat(self.ended_at) if self.ended_at else None

    def is_live(self, now: datetime | None = None) -> bool:
        return self.end is None and (now or utcnow()) >= self.start

    def elapsed_hours(self, now: datetime | None = None) -> float:
        now = now or utcnow()
        return hours((min(now, self.end) if self.end else now - self.start).total_seconds())

    def to_public(self, now: datetime | None = None) -> dict:
        return {
            "account": self.account,
            "kernel_ref": self.kernel_ref,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_hours": self.elapsed_hours(now),
            "live": self.is_live(now),
            "stop_reason": self.stop_reason,
        }


@dataclass
class AccountUsage:
    account: str
    week_started_at: str
    used_hours: float = 0.0
    live_session: dict = field(default_factory=dict)

    def remaining_hours(self, limit: float) -> float:
        return max(0.0, limit - self.used_hours)


class Ledger:
    """Append-only session history, summarised into per-account weekly usage."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._sessions: list[Session] = []
        self._load()

    # ------------------------------------------------------------ persistence

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            raw = {"sessions": []}
        self._sessions = [Session(**s) for s in raw.get("sessions", [])]

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"sessions": [asdict(s) for s in self._sessions]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)

    # ------------------------------------------------------------------ writes

    def open_session(self, account: str, kernel_ref: str, now: datetime | None = None) -> Session:
        now = now or utcnow()
        with self._lock:
            session = Session(account=account, kernel_ref=kernel_ref, started_at=iso(now))
            self._sessions.append(session)
            self._flush()
            return session

    def close_session(
        self,
        session: Session,
        reason: str,
        now: datetime | None = None,
        billed_hours: float | None = None,
    ) -> Session:
        now = now or utcnow()
        with self._lock:
            session.ended_at = iso(now)
            session.stop_reason = reason
            if billed_hours is not None:
                session.billed_hours = billed_hours
            self._flush()
            return session

    def find_live(self, account: str) -> Session | None:
        for session in reversed(self._sessions):
            if session.account == account and session.is_live():
                return session
        return None

    # ------------------------------------------------------------------- reads

    def reconcile(
        self, now: datetime | None = None, max_session_hours: float = 12.0
    ) -> list[Session]:
        """Close sessions left 'live' by an unclean shutdown.

        Without this, a crashed orchestrator would inflate an account's weekly usage
        forever (a live session accrues until `now`), permanently locking it out.
        """
        now = now or utcnow()
        cutoff = now - timedelta(hours=max_session_hours)
        stale = [s for s in self._sessions if s.is_live(now) and s.start < cutoff]
        for session in stale:
            log_message = (
                f"reconciling abandoned session for {session.account} "
                f"started {session.started_at} (older than {max_session_hours}h)"
            )
            print(log_message, file=sys.stderr)
            session.ended_at = iso(session.start + timedelta(hours=max_session_hours))
            session.stop_reason = "reconciled: no pool owned this session"
        if stale:
            self._flush()
        return stale

    def sessions(self) -> list[Session]:
        return list(self._sessions)

    def used_hours(
        self,
        account: str | None = None,
        now: datetime | None = None,
    ) -> float:
        now = now or utcnow()
        start, end = week_start(now), week_end(now)
        total = 0.0
        for session in self._sessions:
            if account and session.account != account:
                continue
            session_end = session.end or now
            total += overlap_seconds(max(session.start, start), session_end, start, end)
        return hours(total)

    def usage(self, account: str, now: datetime | None = None) -> AccountUsage:
        now = now or utcnow()
        live = self.find_live(account)
        return AccountUsage(
            account=account,
            week_started_at=iso(week_start(now)),
            used_hours=self.used_hours(account, now),
            live_session=live.to_public(now) if live else {},
        )

    def total_used_hours(self, now: datetime | None = None) -> float:
        return self.used_hours(None, now)


@dataclass
class Budget:
    """Answers the only two questions the pool asks: may I start, and for how long?"""

    session_limit_hours: float
    weekly_limit_hours: float

    def session_remaining_hours(
        self, session: Session | None, now: datetime | None = None
    ) -> float:
        if session is None or not session.is_live(now):
            return self.session_limit_hours
        return max(0.0, self.session_limit_hours - session.elapsed_hours(now))

    def weekly_remaining_hours(
        self,
        ledger: Ledger,
        account: str,
        account_limit: float = 0.0,
        now: datetime | None = None,
    ) -> float:
        limit = account_limit or self.weekly_limit_hours
        return max(0.0, limit - ledger.used_hours(account, now))

    def can_start(
        self,
        ledger: Ledger,
        account: str,
        account_limit: float = 0.0,
        reserve_hours: float = 0.0,
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        remaining = self.weekly_remaining_hours(ledger, account, account_limit, now)
        needed = max(0.0, reserve_hours)
        if remaining <= 0:
            return False, f"weekly quota exhausted ({remaining:.2f}h left)"
        if needed and remaining < needed:
            return False, f"only {remaining:.2f}h of weekly quota left, {needed:.2f}h reserved"
        if ledger.find_live(account) is not None:
            return False, "account already has a live session"
        return True, f"{remaining:.2f}h of weekly quota left"
