"""Tests for the rate limiting middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rcmes_mcp.middleware.rate_limit import RateLimitMiddleware


@pytest.fixture
def rate_limited_app():
    """Create a test app with tight rate limits for fast testing."""
    app = FastAPI()

    rate_limits = {
        "metadata": {"requests": 5, "window_seconds": 60},
        "data": {"requests": 2, "window_seconds": 60},
    }
    route_tiers = {
        "/api/models": "metadata",
        "/api/load-data": "data",
    }

    app.add_middleware(
        RateLimitMiddleware,
        rate_limits=rate_limits,
        route_tiers=route_tiers,
    )

    @app.get("/api/models")
    async def models():
        return {"models": ["ACCESS-CM2"]}

    @app.post("/api/load-data")
    async def load_data():
        return {"dataset_id": "ds_test"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return TestClient(app)


class TestRateLimiting:
    def test_allows_under_limit(self, rate_limited_app):
        for _ in range(5):
            resp = rate_limited_app.get("/api/models")
            assert resp.status_code == 200

    def test_blocks_over_limit(self, rate_limited_app):
        for _ in range(5):
            rate_limited_app.get("/api/models")

        resp = rate_limited_app.get("/api/models")
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        assert "Rate limit exceeded" in resp.json()["detail"]

    def test_different_tiers_independent(self, rate_limited_app):
        # Exhaust metadata limit
        for _ in range(5):
            rate_limited_app.get("/api/models")
        assert rate_limited_app.get("/api/models").status_code == 429

        # Data tier should still work
        resp = rate_limited_app.post("/api/load-data")
        assert resp.status_code == 200

    def test_unmetered_paths_pass_through(self, rate_limited_app):
        # /health is not in route_tiers, so no rate limiting
        for _ in range(100):
            resp = rate_limited_app.get("/health")
            assert resp.status_code == 200

    def test_data_tier_has_lower_limit(self, rate_limited_app):
        # Data tier allows only 2 requests
        for _ in range(2):
            resp = rate_limited_app.post("/api/load-data")
            assert resp.status_code == 200

        resp = rate_limited_app.post("/api/load-data")
        assert resp.status_code == 429
