"""Multi-account credential store and a thin wrapper around the `kaggle` CLI.

Each account gets its own credential directory so several Kaggle logins can coexist
without clobbering each other. The CLI is invoked as a subprocess with the matching
`KAGGLE_CONFIG_DIR` / `KAGGLE_API_TOKEN` in its environment, which keeps per-account
isolation at the process boundary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .naming import slugify

log = logging.getLogger("kaggle_rotate.accounts")

_CRED_KINDS = ("kaggle_json", "access_token")


@dataclass
class Account:
    slug: str
    username: str
    kind: str = "kaggle_json"
    kernel_slug: str = ""
    enabled: bool = True
    label: str = ""
    # Sessions are attributed to this account in the quota ledger.
    weekly_limit_hours: float = 0.0  # 0 = inherit the global rotation limit
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _CRED_KINDS:
            raise ValueError(f"credential kind must be one of {_CRED_KINDS}")
        if not self.kernel_slug:
            self.kernel_slug = "kaggle-rotate-ollama"
        self.kernel_slug = slugify(self.kernel_slug) or "kaggle-rotate-ollama"

    @property
    def ref(self) -> str:
        return f"{self.username}/{self.kernel_slug}"

    def effective_weekly_limit(self, default: float) -> float:
        return self.weekly_limit_hours or default


# Kaggle phrases "this kernel does not exist / is not yours" as several different
# errors, including a permission error for the owner's own deleted kernels.
_GONE_PATTERNS = (
    "cannot access kernel",
    "was denied",
    "not found",
    "does not exist",
    "no such",
)


def kernel_is_gone(error: KaggleError) -> bool:
    blob = f"{error.stdout}\n{error.stderr}".lower()
    return any(pattern in blob for pattern in _GONE_PATTERNS)


class KaggleError(RuntimeError):
    def __init__(self, args: list[str], returncode: int, stdout: str, stderr: str) -> None:
        detail = (stderr or stdout or "").strip()
        super().__init__(f"`{' '.join(args)}` failed ({returncode}): {detail}")
        self.args_list = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class AccountStore:
    """Registry + credential files for every configured Kaggle account."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.root = config.config_root / "accounts"
        self.registry_path = config.config_root / "accounts.json"

    # ---------------------------------------------------------------- registry

    def load(self) -> dict[str, Account]:
        if not self.registry_path.exists():
            return {}
        raw = json.loads(self.registry_path.read_text())
        return {name: Account(**data) for name, data in raw.items()}

    def save(self, accounts: dict[str, Account]) -> None:
        self.config.config_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.config.config_root, 0o700)
        payload = {name: asdict(acc) for name, acc in accounts.items()}
        self.registry_path.write_text(json.dumps(payload, indent=2) + "\n")
        os.chmod(self.registry_path, 0o600)

    # ------------------------------------------------------------- credentials

    def account_dir(self, slug: str) -> Path:
        return self.root / slug

    def install_credentials(self, slug: str, kind: str, source: Path | str) -> Account:
        """Copy `source` (a kaggle.json or an access_token file) into the account dir."""
        if kind not in _CRED_KINDS:
            raise ValueError(f"credential kind must be one of {_CRED_KINDS}")
        src = Path(source).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"credential file not found: {src}")
        raw = src.read_text().strip()
        if kind == "kaggle_json":
            data = json.loads(raw)
            username = str(data.get("username", "")).strip()
            if not username or not data.get("key"):
                raise ValueError(f"{src} is not a valid kaggle.json (need 'username' and 'key')")
        else:
            username = _username_from_token(raw) or slug

        target_dir = self.account_dir(slug)
        target_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(target_dir, 0o700)
        target = target_dir / ("kaggle.json" if kind == "kaggle_json" else "access_token")
        target.write_text(raw if raw.endswith("\n") else raw + "\n")
        os.chmod(target, 0o600)
        return Account(slug=slug, username=username, kind=kind)

    def remove(self, slug: str, *, purge_credentials: bool = False) -> None:
        accounts = self.load()
        accounts.pop(slug, None)
        self.save(accounts)
        if purge_credentials:
            shutil.rmtree(self.account_dir(slug), ignore_errors=True)

    # --------------------------------------------------------------- execution

    def env_for(self, account: Account) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("KAGGLE_USERNAME", None)
        env.pop("KAGGLE_KEY", None)
        directory = self.account_dir(account.slug)
        if account.kind == "kaggle_json":
            env["KAGGLE_CONFIG_DIR"] = str(directory)
        else:
            env.pop("KAGGLE_CONFIG_DIR", None)
            token = directory / "access_token"
            if token.exists():
                env["KAGGLE_API_TOKEN"] = token.read_text().strip()
        return env


def _username_from_token(token: str) -> str | None:
    """Best-effort: kaggle tokens look like `<label>_<userid>_<secret>`."""
    parts = token.split("_")
    return parts[1] if len(parts) >= 3 and parts[1].isdigit() else None


@dataclass
class StatusResult:
    raw: str
    state: str  # RUNNING | COMPLETE | ERROR | CANCELLED | DEADLINE_EXCEEDED | SUBMITTED | UNKNOWN

    @property
    def terminal(self) -> bool:
        return self.state in {
            "COMPLETE",
            "ERROR",
            "CANCELLED",
            "DEADLINE_EXCEEDED",
        }

    @property
    def alive(self) -> bool:
        return self.state in {"RUNNING", "SUBMITTED", "STARTING"}

    @property
    def failed(self) -> bool:
        return self.state in {"ERROR", "CANCELLED", "DEADLINE_EXCEEDED"}

    @property
    def known(self) -> bool:
        """False when the probe failed or timed out.

        This distinction matters: an unknown status is not evidence that the kernel
        stopped, and treating it as such is how a running kernel gets orphaned.
        """
        return self.state != "UNKNOWN"


TERMINAL_STATES = frozenset({"COMPLETE", "ERROR", "CANCELLED", "DEADLINE_EXCEEDED"})

_STATE_ORDER = (
    "DEADLINE_EXCEEDED",
    "CANCELLED",
    "COMPLETE",
    "ERROR",
    "RUNNING",
    "SUBMITTED",
    "STARTING",
    "INACTIVE",
    "UNKNOWN",
)


def parse_status(text: str) -> StatusResult:
    upper = text.upper()
    for state in _STATE_ORDER:
        if re.search(rf"\b{state}\b", upper):
            return StatusResult(raw=text.strip(), state=state)
    return StatusResult(raw=text.strip(), state="UNKNOWN")


@dataclass
class KaggleCLI:
    config: Config
    store: AccountStore

    def run(
        self,
        account: Account,
        *args: str,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        argv = [self.config.kaggle.command, *args]
        try:
            proc = subprocess.run(
                argv,
                env=self.store.env_for(account),
                capture_output=True,
                text=True,
                timeout=timeout or self.config.kaggle.cli_timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise KaggleError(
                argv, 127, "", f"{self.config.kaggle.command} not found on PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise KaggleError(argv, 124, "", f"timed out after {exc.timeout}s") from exc
        if check and proc.returncode != 0:
            raise KaggleError(argv, proc.returncode, proc.stdout, proc.stderr)
        return proc

    # ------------------------------------------------------------- operations

    def push(self, account: Account, directory: Path, *, accelerator: str, timeout_s: int) -> str:
        return self.run(
            account,
            "kernels",
            "push",
            "-p",
            str(directory),
            "--accelerator",
            accelerator,
            "-t",
            str(timeout_s),
        ).stdout

    def status(self, account: Account) -> StatusResult:
        return parse_status(self.run(account, "kernels", "status", account.ref).stdout)

    async def astatus(self, account: Account, timeout: float | None = None) -> StatusResult:
        """Async status probe.

        The Kaggle API can take tens of seconds to answer (or hang outright). Calling
        the blocking CLI from the event loop freezes the proxy for that whole window,
        which reads to the user as the API being down. Every pool-side call goes
        through here so the loop keeps serving traffic.
        """
        argv = [self.config.kaggle.command, "kernels", "status", account.ref]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                env=self.store.env_for(account),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except (OSError, ValueError):
            return parse_status("kaggle command unavailable")
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout or self.config.kaggle.cli_timeout_seconds,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return parse_status(
                f"status probe timed out after {timeout or self.config.kaggle.cli_timeout_seconds}s"
            )
        return parse_status(stdout.decode("utf-8", "replace"))

    async def adelete(self, account: Account, timeout: float | None = None) -> bool:
        """Delete a kernel. Returns True only if Kaggle confirmed it.

        Deleting is the only dependable stop, so a silent failure here means GPU time
        keeps being billed. Never discard the exit code or stderr.
        """
        argv = [self.config.kaggle.command, "kernels", "delete", account.ref, "--yes"]
        limit = timeout or self.config.kaggle.delete_timeout_seconds
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                env=self.store.env_for(account),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except (OSError, ValueError) as exc:
            log.error("could not run %s: %s", " ".join(argv), exc)
            return False

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=limit)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            log.error("%s timed out after %ss", " ".join(argv), limit)
            return False

        output = stdout.decode("utf-8", "replace").strip()
        if proc.returncode == 0:
            log.info("%s deleted (%s)", account.ref, output or "no output")
            return True

        log.error(
            "%s failed (exit %s): %s",
            " ".join(argv),
            proc.returncode,
            output[:400] or "no output",
        )
        return False

    def delete(self, account: Account) -> bool:
        """Delete a kernel. Returns False if Kaggle refused or the kernel is absent.

        Kaggle answers 403 for a kernel that is already gone, so a non-zero exit here
        does not necessarily mean something is still running. Verify with `status`.
        """
        proc = self.run(account, "kernels", "delete", account.ref, "--yes", check=False)
        if proc.returncode == 0:
            log.info("%s deleted (%s)", account.ref, proc.stdout.strip() or "no output")
            return True
        log.error(
            "%s delete failed (exit %s): %s",
            account.ref,
            proc.returncode,
            (proc.stderr or proc.stdout).strip()[:400] or "no output",
        )
        return False

    def files(self, account: Account, pattern: str = "") -> str:
        args = ["kernels", "files", account.ref, "--page-size", "200"]
        return self.run(account, *args).stdout

    def output(self, account: Account, pattern: str, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        args = ["kernels", "output", account.ref, "-p", str(destination), "-o", "-q"]
        if pattern:
            args += ["--file-pattern", pattern]
        self.run(account, *args, timeout=300, check=False)
        return destination

    def verify(self, account: Account) -> StatusResult:
        """Cheap auth probe: listing own kernels fails fast on bad credentials."""
        self.run(account, "kernels", "list", "--page-size", "1")
        return StatusResult(raw="ok", state="COMPLETE")
