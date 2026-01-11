"""
Caching Utilities

Provides caching for computed results to avoid redundant computation.
Supports both in-memory caching and persistent storage.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import xarray as xr


@dataclass
class CacheEntry:
    """Represents a cached computation result."""

    key: str
    created_at: datetime
    expires_at: datetime
    file_path: str | None = None
    metadata: dict | None = None


class ResultCache:
    """
    Cache for computed climate analysis results.

    Stores results as temporary NetCDF files for datasets,
    or JSON for scalar/dict results.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        default_ttl_hours: int = 24,
    ):
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "rcmes_mcp_cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self._entries: dict[str, CacheEntry] = {}

    def _compute_key(self, *args: Any, **kwargs: Any) -> str:
        """Compute a cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def get(self, key: str) -> xr.Dataset | dict | None:
        """Retrieve a cached result."""
        if key not in self._entries:
            return None

        entry = self._entries[key]

        # Check expiration
        if datetime.now() > entry.expires_at:
            self.delete(key)
            return None

        # Load from file
        if entry.file_path:
            file_path = Path(entry.file_path)
            if file_path.suffix == ".nc":
                return xr.open_dataset(file_path)
            elif file_path.suffix == ".json":
                with open(file_path) as f:
                    return json.load(f)

        return None

    def set(
        self,
        key: str,
        value: xr.Dataset | xr.DataArray | dict,
        ttl: timedelta | None = None,
        metadata: dict | None = None,
    ) -> CacheEntry:
        """Store a result in the cache."""
        if ttl is None:
            ttl = self.default_ttl

        now = datetime.now()
        expires_at = now + ttl

        # Determine file path and save
        if isinstance(value, (xr.Dataset, xr.DataArray)):
            file_path = self.cache_dir / f"{key}.nc"
            value.to_netcdf(file_path)
        elif isinstance(value, dict):
            file_path = self.cache_dir / f"{key}.json"
            with open(file_path, "w") as f:
                json.dump(value, f, default=str)
        else:
            raise TypeError(f"Cannot cache type {type(value)}")

        entry = CacheEntry(
            key=key,
            created_at=now,
            expires_at=expires_at,
            file_path=str(file_path),
            metadata=metadata,
        )
        self._entries[key] = entry

        return entry

    def delete(self, key: str) -> bool:
        """Delete a cached entry."""
        if key not in self._entries:
            return False

        entry = self._entries.pop(key)
        if entry.file_path:
            try:
                Path(entry.file_path).unlink()
            except FileNotFoundError:
                pass

        return True

    def clear(self) -> int:
        """Clear all cached entries. Returns number of entries cleared."""
        count = len(self._entries)
        for key in list(self._entries.keys()):
            self.delete(key)
        return count

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        now = datetime.now()
        expired = [key for key, entry in self._entries.items() if now > entry.expires_at]
        for key in expired:
            self.delete(key)
        return len(expired)


# Global cache instance
result_cache = ResultCache()


def cached(ttl_hours: int = 24):
    """
    Decorator for caching function results.

    Usage:
        @cached(ttl_hours=12)
        def expensive_computation(data, param):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Compute cache key
            key = result_cache._compute_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = result_cache.get(key)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            result_cache.set(key, result, ttl=timedelta(hours=ttl_hours))

            return result

        return wrapper

    return decorator
