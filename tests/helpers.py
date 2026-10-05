"""Shared helpers: run throwaway uvicorn servers on ephemeral ports."""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from typing import Any

import uvicorn


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _Server(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        return


@asynccontextmanager
async def serve(app: Any, port: int | None = None):
    port = port or free_port()
    server = _Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    )
    task = asyncio.create_task(server.serve())
    deadline = asyncio.get_running_loop().time() + 20
    while not server.started and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    assert server.started, "server did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=10)


async def settle(seconds: float = 0.15) -> None:
    await asyncio.sleep(seconds)
