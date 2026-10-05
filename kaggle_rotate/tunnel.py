"""Locate (or fetch) the `cloudflared` binary and run a quick tunnel through it.

Two tunnels are used: one on the local machine to publish the control relay so Kaggle
can call back, and one inside each kernel to publish the model endpoint. Neither needs
a Cloudflare account, but neither URL is stable or authenticated, which is why the
relay is token-guarded and clients should keep using the local proxy.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

import httpx

QUICK_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

ASSETS = {
    ("linux", "x86_64"): "cloudflared-linux-amd64",
    ("linux", "aarch64"): "cloudflared-linux-arm64",
    ("darwin", "x86_64"): "cloudflared-darwin-amd64",
    ("darwin", "arm64"): "cloudflared-darwin-arm64.tgz",
}
BASE_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download"


class TunnelError(RuntimeError):
    pass


@dataclass
class Tunnel:
    url: str
    process: asyncio.subprocess.Process

    async def stop(self) -> None:
        if self.process.returncode is not None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=15)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()


def binary_candidates(cache_dir: Path) -> list[Path]:
    return [cache_dir / "cloudflared", cache_dir / "cloudflared.exe"]


def find_cloudflared(explicit: str = "", cache_dir: Path | None = None) -> str | None:
    if explicit:
        path = Path(explicit).expanduser()
        return str(path) if path.is_file() else None
    found = shutil.which("cloudflared")
    if found:
        return found
    if cache_dir:
        for candidate in binary_candidates(cache_dir):
            if candidate.is_file():
                return str(candidate)
    return None


async def ensure_cloudflared(cache_dir: Path, explicit: str = "") -> str:
    found = find_cloudflared(explicit, cache_dir)
    if found:
        return found

    key = (platform.system().lower(), platform.machine().lower())
    asset = ASSETS.get(key)
    if asset is None:
        raise TunnelError(
            f"no cloudflared build for {key}; install it manually or set kernel.cloudflared_path"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "cloudflared"
    url = f"{BASE_URL}/{asset}"
    async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            tmp = target.with_suffix(".part")
            with tmp.open("wb") as fh:
                async for chunk in response.aiter_bytes():
                    fh.write(chunk)

    if asset.endswith(".tgz"):
        import tarfile

        with tarfile.open(target.with_suffix(".part")) as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith("cloudflared"))
            member.name = "cloudflared"
            archive.extract(member, path=cache_dir)
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(target)

    target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(target)


async def start_quick_tunnel(binary: str, target: str, timeout: float = 90.0) -> Tunnel:
    """Start a quick tunnel to `target` and return once Cloudflare hands out a URL."""
    process = await asyncio.create_subprocess_exec(
        binary,
        "tunnel",
        "--no-autoupdate",
        "--protocol",
        "http2",
        "--edge-ip-version",
        "4",
        "--url",
        target,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "NO_COLOR": "1"},
    )

    assert process.stdout is not None
    try:

        async def read() -> str:
            while True:
                line = await process.stdout.readline()
                if not line:
                    return ""
                text = line.decode("utf-8", "replace")
                match = QUICK_TUNNEL_RE.search(text)
                if match:
                    return match.group(0)

        url = await asyncio.wait_for(read(), timeout=timeout)
    except TimeoutError as exc:
        process.kill()
        raise TunnelError(f"cloudflared produced no URL within {timeout}s") from exc

    if not url:
        process.kill()
        raise TunnelError("cloudflared exited before allocating a URL")
    return Tunnel(url=url, process=process)


async def wait_reachable(url: str, timeout: float = 60.0, interval: float = 2.0) -> bool:
    """Poll a public URL until it answers, so we never hand a dead relay to a kernel."""
    deadline = asyncio.get_running_loop().time() + timeout
    async with httpx.AsyncClient(timeout=10.0) as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                response = await client.get(url)
                if response.status_code < 500:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(interval)
    return False
