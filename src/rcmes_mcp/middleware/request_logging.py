"""
Request/response logging middleware for the RCMES API.

Logs every API request with method, path, status code, and duration.
Skips static asset requests and health checks to reduce noise.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("rcmes.api.requests")

# Paths to skip logging (static assets, favicon, etc.)
_SKIP_PREFIXES = ("/assets/", "/favicon", "/_app/")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every API request with timing information."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip static asset noise
        if any(path.startswith(p) for p in _SKIP_PREFIXES):
            return await call_next(request)

        method = request.method
        client_ip = request.client.host if request.client else "unknown"
        request_id = uuid.uuid4().hex[:8]

        start = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            status = response.status_code

            log_level = logging.INFO
            if status >= 500:
                log_level = logging.ERROR
            elif status >= 400:
                log_level = logging.WARNING

            # Skip health checks at INFO level (log at DEBUG only)
            if path == "/api/health" and status == 200:
                log_level = logging.DEBUG

            logger.log(
                log_level,
                f"{method} {path} → {status} ({duration_ms}ms)",
                extra={
                    "method": method,
                    "path": path,
                    "status": status,
                    "duration_ms": duration_ms,
                    "client_ip": client_ip,
                    "request_id": request_id,
                },
            )
            return response

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.exception(
                f"{method} {path} → 500 ({duration_ms}ms) {type(exc).__name__}",
                extra={
                    "method": method,
                    "path": path,
                    "status": 500,
                    "duration_ms": duration_ms,
                    "client_ip": client_ip,
                    "request_id": request_id,
                    "error": str(exc),
                },
            )
            raise
