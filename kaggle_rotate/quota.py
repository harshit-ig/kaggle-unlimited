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
        # A session cannot have run past its own end time, but it also cannot have run
        # for a negative span if the clock moved. Clamp both ends so the arithmetic is
        # always a timedelta.
        end = min(now, self.end) if self.end else now
        return hours(max(0.0, (end - self.start).total_seconds()))

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
        kaggle_remaining: float | None = None,
        kaggle_used: float | None = None,
    ) -> tuple[bool, str]:
        """May this account start a session?

        `kaggle_remaining` is Kaggle's own "hours left this week" when we have one, and
        it is preferred because it counts GPU time this tool did not start; the ledger
        only knows about its own sessions and therefore under-counts. The fallback is
        the ledger, which fails open on purpose: a quota-API outage must not strand a
        healthy account behind a working endpoint. That choice can overshoot a real cap,
        so callers log whenever the fallback is used.

        A per-account `account_limit` overrides Kaggle's total, so an operator can cap an
        account below what Kaggle would allow. That needs `kaggle_used` rather than
        `kaggle_remaining`, because remaining-under-a-cap is derived from used time --
        capping "what's left" directly would silently ignore the cap whenever Kaggle's
        own allowance was the smaller number.
        """
        local = self.weekly_remaining_hours(ledger, account, account_limit, now)
        if account_limit and kaggle_used is not None:
            remaining = max(0.0, account_limit - kaggle_used)
        elif kaggle_remaining is not None:
            remaining = kaggle_remaining
        else:
            remaining = local

        needed = max(0.0, reserve_hours)
        if remaining <= 0:
            source = "kaggle" if kaggle_remaining is not None else "ledger"
            return False, f"weekly quota exhausted ({remaining:.2f}h left, from {source})"
        if needed and remaining < needed:
            return False, f"only {remaining:.2f}h of weekly quota left, {needed:.2f}h reserved"
        if ledger.find_live(account) is not None:
            return False, "account already has a live session"
        return True, f"{remaining:.2f}h of weekly quota left"


@dataclass
class QuotaSnapshot:
    """Kaggle's own accelerator accounting for one account.

    This is ground truth: it includes GPU time this tool did not start, which is why
    the local ledger alone is not safe for budget decisions. Measured on a real account,
    the ledger under-counted 5.03 billed hours as 2.72h of wall time.

    `fetched_at` is stamped on construction unless a caller supplies one, so a snapshot
    is always timestamped even when it is built by hand in a test or a parse.
    """

    account: str
    used_hours: float
    remaining_hours: float
    total_hours: float
    # When Kaggle says the weekly counter resets. Preferred over assuming Saturday 00:00.
    refresh_at: str = ""
    fetched_at: float = field(default_factory=lambda: utcnow().timestamp())
    # How old this reading is. Recomputed on cache hits; a hand-built snapshot is fresh.
    age_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.used_hours < 0:
            self.used_hours = 0.0
        # Kaggle can report a used total above the allowance (e.g. a shortened week), so
        # clamp `remaining` rather than let it read negative, but keep `used` intact so
        # an overage is still visible.
        self.remaining_hours = max(0.0, self.remaining_hours)

    def to_public(self) -> dict:
        return {
            "account": self.account,
            "used_hours": round(self.used_hours, 4),
            "remaining_hours": round(self.remaining_hours, 4),
            "total_hours": round(self.total_hours, 4),
            "refresh_at": self.refresh_at,
            "age_seconds": round(self.age_seconds, 1),
        }


class QuotaUnavailable(Exception):
    """Kaggle's quota API could not be read. Callers fall back to the local ledger."""


def parse_quota_csv(text: str, account: str) -> QuotaSnapshot | None:
    """Parse `kaggle quota --csv`.

    The CSV is the stable contract here rather than the protobuf API: it is one
    subprocess either way, and it does not add a kaggle-python dependency to the
    subprocess path. Header is `resource,used,remaining,total,refreshAt`.
    """
    gpu: list[str] | None = None
    refresh = ""
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        if parts[0].lower() == "resource":
            continue
        if parts[0].upper() == "GPU" and gpu is None:
            gpu = parts

    def as_hours(value: str) -> float | None:
        value = value.strip().upper()
        if not value.endswith("H"):
            return None
        try:
            return float(value[:-1])
        except ValueError:
            return None

    if gpu is None or len(gpu) < 4:
        return None
    used = as_hours(gpu[1])
    remaining = as_hours(gpu[2])
    total = as_hours(gpu[3])
    if used is None or remaining is None or total is None:
        return None
    if len(gpu) > 4 and gpu[4]:
        refresh = gpu[4]
    return QuotaSnapshot(
        account=account,
        used_hours=used,
        remaining_hours=remaining,
        total_hours=total,
        refresh_at=refresh,
    )


class QuotaCache:
    """Per-account cache of Kaggle's quota numbers, with an explicit failure signal.

    The pool ticks every 10s and a quota read costs ~0.6s, so an uncached read per tick
    per account would be a meaningful tax on the loop for a number that moves slowly.
    Five minutes is short against a 30h budget and long enough to keep the cost flat.
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, QuotaSnapshot] = {}
        self._lock = threading.Lock()
        self.reads = 0
        self.hits = 0
        self.failures = 0

    def cached(self, account: str) -> QuotaSnapshot | None:
        with self._lock:
            entry = self._entries.get(account)
            if entry is None:
                return None
            age = utcnow().timestamp() - entry.fetched_at
            if age > self.ttl_seconds:
                return None
            self.hits += 1
            entry.age_seconds = age
            return entry

    def store(self, snapshot: QuotaSnapshot) -> None:
        with self._lock:
            self._entries[snapshot.account] = snapshot

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"reads": self.reads, "hits": self.hits, "failures": self.failures}
