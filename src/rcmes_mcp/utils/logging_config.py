"""
Centralized logging configuration for RCMES-MCP.

Provides structured JSON logging for production and human-readable
logging for development. Call configure_logging() once at startup.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any extras set via logger.info("msg", extra={...})
        for key in ("method", "path", "status", "duration_ms", "client_ip",
                     "dataset_id", "variable", "model", "scenario",
                     "tool", "tier", "error", "request_id"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


class ReadableFormatter(logging.Formatter):
    """Compact human-readable format for development."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname[0]  # I, W, E, D
        msg = record.getMessage()

        # Append key extras inline
        extras = []
        for key in ("method", "path", "status", "duration_ms", "dataset_id", "tool"):
            val = getattr(record, key, None)
            if val is not None:
                extras.append(f"{key}={val}")
        if extras:
            msg = f"{msg}  [{', '.join(extras)}]"

        base = f"{ts} {level} [{record.name}] {msg}"
        if record.exc_info and record.exc_info[1]:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str | None = None) -> None:
    """Set up logging for the entire application.

    Args:
        level: Override log level. Defaults to RCMES_LOG_LEVEL env var, or INFO.
    """
    log_level = (level or os.environ.get("RCMES_LOG_LEVEL", "INFO")).upper()
    log_format = os.environ.get("RCMES_LOG_FORMAT", "readable").lower()

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # Remove any existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ReadableFormatter())
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("botocore", "s3fs", "urllib3", "aiobotocore", "fsspec",
                   "s3transfer", "asyncio", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
