"""
Session Management

Manages user sessions and dataset references. Datasets are loaded lazily
and cached for the duration of the session to avoid repeated I/O.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import xarray as xr


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset."""

    dataset_id: str
    source: str  # "nex-gddp-cmip6", "nex-dcp30", "rcmed", etc.
    variable: str
    model: str | None = None
    scenario: str | None = None
    time_range: tuple[str, str] | None = None
    spatial_bounds: dict[str, float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""


class SessionManager:
    """
    Manages datasets loaded during a session.

    Datasets are stored with unique IDs and can be referenced in
    subsequent tool calls. This enables chaining operations like:
    load -> subset -> rebin -> analyze -> visualize
    """

    def __init__(self, ttl_hours: int = 2):
        self._datasets: dict[str, xr.Dataset | xr.DataArray] = {}
        self._metadata: dict[str, DatasetInfo] = {}
        self._ttl = timedelta(hours=ttl_hours)

    def generate_id(self, prefix: str = "ds") -> str:
        """Generate a unique dataset ID."""
        unique = uuid.uuid4().hex[:8]
        return f"{prefix}_{unique}"

    def store(
        self,
        data: xr.Dataset | xr.DataArray,
        source: str,
        variable: str,
        model: str | None = None,
        scenario: str | None = None,
        description: str = "",
        **kwargs: Any,
    ) -> str:
        """
        Store a dataset and return its ID.

        Args:
            data: xarray Dataset or DataArray to store
            source: Data source identifier
            variable: Climate variable name
            model: Climate model name (if applicable)
            scenario: Emissions scenario (if applicable)
            description: Human-readable description
            **kwargs: Additional metadata

        Returns:
            Unique dataset ID for referencing in subsequent operations
        """
        dataset_id = self.generate_id()

        # Extract time range if present
        time_range = None
        if "time" in data.dims:
            times = data.time.values
            time_range = (str(times[0])[:10], str(times[-1])[:10])

        # Extract spatial bounds if present
        spatial_bounds = None
        if "lat" in data.dims and "lon" in data.dims:
            spatial_bounds = {
                "lat_min": float(data.lat.min()),
                "lat_max": float(data.lat.max()),
                "lon_min": float(data.lon.min()),
                "lon_max": float(data.lon.max()),
            }

        self._datasets[dataset_id] = data
        self._metadata[dataset_id] = DatasetInfo(
            dataset_id=dataset_id,
            source=source,
            variable=variable,
            model=model,
            scenario=scenario,
            time_range=time_range,
            spatial_bounds=spatial_bounds,
            description=description,
        )

        return dataset_id

    def get(self, dataset_id: str) -> xr.Dataset | xr.DataArray:
        """Retrieve a dataset by ID."""
        if dataset_id not in self._datasets:
            raise KeyError(f"Dataset '{dataset_id}' not found. Available: {list(self._datasets.keys())}")
        return self._datasets[dataset_id]

    def get_metadata(self, dataset_id: str) -> DatasetInfo:
        """Retrieve metadata for a dataset."""
        if dataset_id not in self._metadata:
            raise KeyError(f"Dataset '{dataset_id}' not found.")
        return self._metadata[dataset_id]

    def list_datasets(self) -> list[dict]:
        """List all datasets in the session."""
        return [
            {
                "id": info.dataset_id,
                "source": info.source,
                "variable": info.variable,
                "model": info.model,
                "scenario": info.scenario,
                "time_range": info.time_range,
                "spatial_bounds": info.spatial_bounds,
                "description": info.description,
            }
            for info in self._metadata.values()
        ]

    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset from the session."""
        if dataset_id in self._datasets:
            del self._datasets[dataset_id]
            del self._metadata[dataset_id]
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove expired datasets. Returns count of removed datasets."""
        now = datetime.now()
        expired = [
            ds_id
            for ds_id, info in self._metadata.items()
            if now - info.created_at > self._ttl
        ]
        for ds_id in expired:
            self.delete(ds_id)
        return len(expired)

    def clear(self) -> None:
        """Clear all datasets from the session."""
        self._datasets.clear()
        self._metadata.clear()


# Global session manager instance
session_manager = SessionManager()
