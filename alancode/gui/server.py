"""FastAPI server for the Alan Code GUI.

Serves the browser SPA and provides a WebSocket endpoint.
Runs as an asyncio background task alongside the session loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from alancode.gui.gui_ui import GUIUI

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 8420
MAX_PORT_ATTEMPTS = 10


def _find_available_port(start: int = DEFAULT_PORT, attempts: int = MAX_PORT_ATTEMPTS) -> int:
    """Find an available port starting from *start*.

    Uses ``SO_REUSEADDR`` so that ports in TCP ``TIME_WAIT`` state
    (from a recently killed server) are considered available.
    """
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No available port found in range {start}-{start + attempts - 1}"
    )


def _cwd_url_segment(cwd: str) -> str:
    """Last component of the path, used as URL segment."""
    return Path(cwd).name or "alan"


def create_gui_app(gui_ui: GUIUI, cwd: str = "") -> FastAPI:
    """Create the FastAPI application for the GUI."""
    app = FastAPI(title="Alan Code GUI", docs_url=None, redoc_url=None)
    project_name = _cwd_url_segment(cwd)

    # ── Session info endpoint ─────────────────────────────────────────

    @app.get("/api/session")
    async def session_info():
        agent = gui_ui._agent
        return {
            "session_id": agent.session_id if agent else "",
            "session_name": agent._session.session_name if agent else "",
            "project": project_name,
            "cwd": cwd,
            "model": agent._model if agent else "",
        }

    # ── WebSocket endpoint ────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        gui_ui.add_connection(websocket)

        # Send history for replay
        await gui_ui.send_history(websocket)

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Ignored non-JSON WebSocket frame (%s): %.200r",
                        exc, raw,
                    )
                    continue
                await gui_ui.handle_ws_message(data)

        except WebSocketDisconnect:
            gui_ui.remove_connection(websocket)
        except Exception:
            gui_ui.remove_connection(websocket)

    # ── Static files ──────────────────────────────────────────────────

    @app.get(f"/{project_name}/")
    @app.get(f"/{project_name}")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/")
    async def root_redirect():
        return RedirectResponse(url=f"/{project_name}/")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


async def start_gui_server(
    gui_ui: GUIUI,
    cwd: str = "",
    port: int | None = None,
) -> tuple[str, "uvicorn.Server", asyncio.Task]:
    """Start the GUI server as a background asyncio task.

    Returns ``(url, server, task)`` so the caller can shut it down cleanly.
    """
    import uvicorn

    if port is None:
        port = _find_available_port()

    app = create_gui_app(gui_ui, cwd=cwd)
    project_name = _cwd_url_segment(cwd)
    url = f"http://localhost:{port}/{project_name}/"

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        # Don't hang forever waiting for websockets to close on shutdown.
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)

    logger.info("GUI server starting at %s", url)
    task = asyncio.create_task(server.serve())

    # Wait for the server to bind
    for _ in range(50):
        await asyncio.sleep(0.1)
        if server.started:
            break
    else:
        logger.warning("GUI server did not start within 5 seconds")

    return url, server, task
