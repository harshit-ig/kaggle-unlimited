"""Typed configuration loaded from TOML, with defaults for everything."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAME = "config.toml"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 8317
    # Seconds a client request may wait for the pool to hand it a usable upstream.
    connect_grace_seconds: float = 90.0
    # Retried once against the standby if the active upstream dies mid-request.
    retry_on_standby: bool = True
    # Retained so the previous upstream is not dropped while streams are open.
    drain_grace_seconds: float = 30.0


@dataclass
class RelayConfig:
    host: str = "127.0.0.1"
    port: int = 8318
    # Publish the relay through a Cloudflare quick tunnel so Kaggle can call back.
    expose: bool = True
    tunnel_start_timeout: float = 90.0
    # Extra hostnames/urls to use instead of a quick tunnel (e.g. a named tunnel you
    # already expose). If set, `expose` is ignored.
    public_url: str = ""
    # How long the relay waits without a pulse from the pool before it tells kernels
    # the driver is gone. A relay can outlive its pool (orphaned cloudflared), and a
    # kernel that keeps getting replies from a driverless relay would never notice it
    # had been abandoned.
    driver_grace_seconds: float = 120.0


@dataclass
class RotationConfig:
    session_limit_hours: float = 12.0
    weekly_limit_hours: float = 30.0
    # Reserve GPU time for the *next* boot so a warmed session always has headroom.
    boot_reserve_hours: float = 1.5
    # Start the next account this long before the active one hits its session cap.
    prewarm_lead_minutes: float = 120.0
    # Stop launching a session if it would begin within this of a weekly cap.
    weekly_safety_minutes: float = 60.0
    max_active_sessions: int = 1
    # Grace period between cutting clients over and telling the old kernel to exit.
    drain_grace_seconds: float = 60.0
    boot_timeout_seconds: float = 2400.0
    # A kernel that has not heartbeated for this long is considered dead.
    heartbeat_grace_seconds: float = 240.0
    # 0 disables idle shutdown (saves quota when nobody is calling the endpoint).
    idle_stop_minutes: float = 0.0
    tick_seconds: float = 10.0
    # Backoff after a failed launch (bad credentials, kernel push refused, ...).
    launch_retry_seconds: float = 120.0
    launch_retry_max_seconds: float = 1800.0


@dataclass
class KernelConfig:
    accelerator: str = "NvidiaTeslaT4"
    kernel_slug: str = "kaggle-rotate-llamacpp"
    # Must slugify to kernel_slug; Kaggle derives the kernel slug from the title.
    title: str = ""
    is_private: bool = True
    # Notebook the server is built from. It is also the source of truth for the model,
    # the weights and the context width: there is no config surface for any of those.
    source_notebook: str = "bonsai2_27b_pq2_0_kaggle_llamacpp.ipynb"
    # Notebook self-terminates before Kaggle's hard cap so quota is never wasted.
    max_runtime_minutes: float = 660.0
    shutdown_poll_seconds: float = 20.0
    # How long a kernel tolerates an unreachable relay before self-terminating.
    # This is the backstop that stops a GPU session burning quota when the local
    # pool dies, so keep it well under the session cap.
    orphan_grace_seconds: float = 300.0
    # Locate cloudflared on PATH / cache dir, or an explicit path.
    cloudflared_path: str = ""


@dataclass
class KaggleConfig:
    command: str = "kaggle"
    config_root: str = "~/.config/kaggle-rotate"
    cli_timeout_seconds: float = 60.0
    # `kaggle kernels delete` routinely takes 60-150s. Killing it early means the
    # kernel survives and keeps billing, so this gets its own, generous budget.
    delete_timeout_seconds: float = 300.0


@dataclass
class Config:
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    relay: RelayConfig = field(default_factory=RelayConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    kaggle: KaggleConfig = field(default_factory=KaggleConfig)

    root: Path = field(default_factory=_project_root)
    state_dir: Path = field(default_factory=lambda: Path("run"))

    @property
    def config_root(self) -> Path:
        raw = self.kaggle.config_root
        return Path(raw).expanduser().resolve()

    def resolved_state_dir(self) -> Path:
        path = self.state_dir
        if not path.is_absolute():
            path = self.root / path
        return path

    def resolved_source_notebook(self) -> Path:
        path = Path(self.kernel.source_notebook)
        if not path.is_absolute():
            path = self.root / path
        return path

    def kernel_ref(self, username: str) -> str:
        """The Kaggle kernel this tool owns for `username`.

        Kaggle derives a kernel's slug from its title, and `kernel-metadata.json`'s `id`
        has to match that or the push is rejected. Both the `id` and the title are built
        from `kernel.kernel_slug`, and both are built *here* so they cannot drift apart.
        """
        return f"{username}/{self.kernel.kernel_slug}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "proxy": asdict(self.proxy),
            "relay": asdict(self.relay),
            "rotation": asdict(self.rotation),
            "kernel": asdict(self.kernel),
            "kaggle": asdict(self.kaggle),
            "state_dir": str(self.resolved_state_dir()),
        }


_SECTIONS = {
    "proxy": ProxyConfig,
    "relay": RelayConfig,
    "rotation": RotationConfig,
    "kernel": KernelConfig,
    "kaggle": KaggleConfig,
}


def _build(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown key(s) in [{cls.__name__}]: {', '.join(sorted(unknown))}")
    return cls(**raw)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    candidate = Path(path) if path else _project_root() / DEFAULT_CONFIG_NAME
    if not candidate.exists():
        return cfg
    with candidate.open("rb") as fh:
        raw = tomllib.load(fh)

    top = dict(raw)
    for name, cls in _SECTIONS.items():
        section = top.pop(name, {})
        if section:
            setattr(cfg, name, _build(cls, section))
    state_dir = top.pop("state_dir", None)
    if state_dir:
        cfg.state_dir = Path(state_dir)
    if top:
        raise ValueError(f"unknown top-level key(s): {', '.join(sorted(top))}")
    return cfg
