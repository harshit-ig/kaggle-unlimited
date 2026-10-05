"""Durable pool state, so a restarted orchestrator can still retire orphaned kernels."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SlotState:
    """One account's slot. States move idle -> launching -> booting -> ready ->
    active -> draining -> stopped (or failed)."""

    account: str
    kernel_ref: str = ""
    token: str = ""
    state: str = "idle"
    url: str = ""
    model: str = ""
    started_at: float = 0.0
    last_transition: float = field(default_factory=time.time)
    detail: str = ""
    last_attempt: float = 0.0
    attempts: int = 0
    # Kaggle's own weekly accelerator usage for this account, when the pool has been
    # able to read it. The ledger cannot see GPU time this tool did not start, so these
    # are the figures a budget decision should be based on.
    kaggle_weekly_used_hours: float | None = None
    kaggle_weekly_remaining_hours: float | None = None
    kaggle_quota_refresh_at: str = ""
    # "kaggle" once a real reading is available, otherwise "ledger".
    quota_source: str = "ledger"

    def to_public(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("token", None)
        data["session_age_s"] = round(time.time() - self.started_at, 1) if self.started_at else 0.0
        return data


class PoolState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.slots: dict[str, SlotState] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return
        for account, data in raw.get("slots", {}).items():
            known = {f for f in SlotState.__dataclass_fields__}
            self.slots[account] = SlotState(**{k: v for k, v in data.items() if k in known})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"slots": {name: asdict(slot) for name, slot in self.slots.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)

    def get(self, account: str) -> SlotState:
        if account not in self.slots:
            self.slots[account] = SlotState(account=account)
        return self.slots[account]

    def update(self, account: str, **fields: Any) -> SlotState:
        slot = self.get(account)
        if "state" in fields and fields["state"] != slot.state:
            slot.last_transition = time.time()
        for key, value in fields.items():
            if hasattr(slot, key):
                setattr(slot, key, value)
        self.save()
        return slot

    def live_accounts(self) -> list[str]:
        return [
            name
            for name, slot in self.slots.items()
            if slot.state in {"launching", "booting", "ready", "active", "draining"}
        ]

    def orphans(self) -> list[SlotState]:
        return [
            slot
            for slot in self.slots.values()
            if slot.state in {"launching", "booting", "ready", "active", "draining"} and slot.token
        ]


class PidFile:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> int | None:
        if not self.path.exists():
            return None
        try:
            pid = int(self.path.read_text().strip())
        except ValueError:
            return None
        return pid if pid_is_alive(pid) else None

    def write(self, pid: int | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pid = pid or os.getpid()
        self.path.write_text(f"{pid}\n")

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
