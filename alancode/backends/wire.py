"""Wire audit - record the exact request a backend hands its provider SDK.

The GUI's LLM perspective is rendered from Alan's message list *before*
backend-specific shaping (serialization, cache breakpoints, provider quirks),
so it cannot prove what was actually sent. Set ``ALANCODE_WIRE_LOG`` to a file
path and every outgoing request is appended there as one JSON line.

Disabled by default; when disabled the call is a dict lookup and a return.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

WIRE_LOG_ENV = "ALANCODE_WIRE_LOG"

# Shallow-redacted before serialization: these reach the audit as top-level
# request keys and must never be written to a file the user may share.
REDACTED_KEYS = frozenset({"api_key", "authorization", "headers", "extra_headers"})
REDACTED_PLACEHOLDER = "<redacted>"

_wire_logger: logging.Logger | None = None
_setup_failed = False


def _wire_log() -> logging.Logger | None:
    """Return the audit logger, or None when the audit is off or unusable."""
    global _wire_logger, _setup_failed
    if _wire_logger is not None:
        return _wire_logger
    if _setup_failed:
        return None
    path = os.environ.get(WIRE_LOG_ENV)
    if not path:
        return None
    try:
        handler = logging.FileHandler(path)
    except OSError as exc:
        logger.warning("Cannot open %s=%s for the wire audit: %s", WIRE_LOG_ENV, path, exc)
        _setup_failed = True
        return None
    handler.setFormatter(logging.Formatter("%(message)s"))
    wire = logging.getLogger("alancode.wire")
    wire.addHandler(handler)
    wire.setLevel(logging.DEBUG)
    wire.propagate = False  # a raw JSONL file, never mixed into the app log
    _wire_logger = wire
    logger.info("Wire audit enabled: appending provider requests to %s", path)
    return wire


def log_wire_request(provider: str, request: dict[str, Any]) -> None:
    """Append one outgoing provider request to the wire audit, if enabled.

    Args:
        provider: Transport name, e.g. ``"litellm"`` or ``"anthropic"``.
        request: The keyword arguments about to be passed to the provider SDK.
    """
    wire = _wire_log()
    if wire is None:
        return
    redacted = {
        k: (REDACTED_PLACEHOLDER if k in REDACTED_KEYS else v)
        for k, v in request.items()
    }
    try:
        line = json.dumps({"provider": provider, "request": redacted}, default=str)
    except (TypeError, ValueError) as exc:
        logger.warning("Wire audit could not serialize the %s request: %s", provider, exc)
        return
    wire.debug(line)
