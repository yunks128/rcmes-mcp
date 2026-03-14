"""
In-memory sliding-window rate limiter for the RCMES API.

Uses per-IP request tracking with configurable tiers so that
lightweight metadata lookups have generous limits while expensive
data-loading and chat endpoints are more restricted.
"""

from __future__ import annotations

import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# ---------------------------------------------------------------------------
# Default tier configuration
# ---------------------------------------------------------------------------

RATE_LIMITS: dict[str, dict[str, int]] = {
    "metadata": {"requests": 120, "window_seconds": 60},
    "data":     {"requests": 10,  "window_seconds": 60},
    "analysis": {"requests": 30,  "window_seconds": 60},
    "chat":     {"requests": 20,  "window_seconds": 60},
}

ROUTE_TIERS: dict[str, str] = {
    # Metadata — lightweight lookups
    "/api/models":         "metadata",
    "/api/variables":      "metadata",
    "/api/scenarios":      "metadata",
    "/api/datasets":       "metadata",
    "/api/countries":      "metadata",
    "/api/country-bounds": "metadata",
    "/api/health":         "metadata",
    # Data — expensive S3 access
    "/api/load-data":      "data",
    "/api/download":       "data",
    # Analysis / processing
    "/api/spatial-subset":  "analysis",
    "/api/temporal-subset": "analysis",
    "/api/regrid":          "analysis",
    "/api/convert-units":   "analysis",
    "/api/mask-by-country": "analysis",
    "/api/analyze":         "analysis",
    "/api/etccdi":          "analysis",
    "/api/batch-etccdi":    "analysis",
    "/api/correlation":     "analysis",
    "/api/visualize":       "analysis",
    # Chat — moderate
    "/api/chat/stream":     "chat",
    "/api/chat":            "chat",
}


def _match_tier(path: str, route_tiers: dict[str, str]) -> str | None:
    """Return the tier for a request path, or None if unmetered."""
    # Longest-prefix match so /api/chat/stream beats /api/chat
    best_match = ""
    best_tier = None
    for prefix, tier in route_tiers.items():
        if path.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_tier = tier
    return best_tier


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by (client_ip, tier)."""

    def __init__(self, app, rate_limits=None, route_tiers=None):
        super().__init__(app)
        self.rate_limits = rate_limits or RATE_LIMITS
        self.route_tiers = route_tiers or ROUTE_TIERS
        # {(ip, tier): deque[timestamp]}
        self._windows: dict[tuple[str, str], deque[float]] = {}
        self._request_count = 0

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        tier = _match_tier(path, self.route_tiers)

        # Non-API or unmetered paths pass through
        if tier is None:
            return await call_next(request)

        limit_cfg = self.rate_limits.get(tier)
        if limit_cfg is None:
            return await call_next(request)

        max_requests = limit_cfg["requests"]
        window_sec = limit_cfg["window_seconds"]

        client_ip = request.client.host if request.client else "unknown"
        key = (client_ip, tier)
        now = time.time()

        window = self._windows.setdefault(key, deque())

        # Prune expired entries
        cutoff = now - window_sec
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= max_requests:
            retry_after = int(window[0] + window_sec - now) + 1
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded for {tier} endpoints. "
                    f"Limit: {max_requests} requests per {window_sec}s.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)

        # Periodic cleanup of stale keys (every 1000 requests)
        self._request_count += 1
        if self._request_count % 1000 == 0:
            self._cleanup_stale(now)

        return await call_next(request)

    def _cleanup_stale(self, now: float) -> None:
        """Remove keys whose entries are all expired."""
        stale_keys = []
        for key, window in self._windows.items():
            _, tier = key
            cfg = self.rate_limits.get(tier, {})
            window_sec = cfg.get("window_seconds", 60)
            if not window or window[-1] < now - window_sec:
                stale_keys.append(key)
        for key in stale_keys:
            del self._windows[key]
