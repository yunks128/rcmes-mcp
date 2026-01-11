"""Tests for session management."""

import numpy as np
import pytest
import xarray as xr

from rcmes_mcp.utils.session import DatasetInfo, SessionManager, session_manager


@pytest.fixture
def sample_dataset():
    """Create a sample xarray dataset for testing."""
    times = np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]")
    lats = np.linspace(30, 40, 10)
    lons = np.linspace(-120, -110, 10)

    data = np.random.rand(len(times), len(lats), len(lons))

    ds = xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], data, {"units": "K"}),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    return ds


@pytest.fixture
def manager():
    """Create a fresh session manager for each test."""
    return SessionManager(ttl_hours=1)


class TestSessionManager:
    """Test SessionManager class."""

    def test_generate_id(self, manager):
        """Test ID generation."""
        id1 = manager.generate_id()
        id2 = manager.generate_id()
        assert id1 != id2
        assert id1.startswith("ds_")
        assert id2.startswith("ds_")

    def test_generate_id_custom_prefix(self, manager):
        """Test ID generation with custom prefix."""
        id1 = manager.generate_id(prefix="test")
        assert id1.startswith("test_")

    def test_store_and_get(self, manager, sample_dataset):
        """Test storing and retrieving a dataset."""
        dataset_id = manager.store(
            data=sample_dataset,
            source="test",
            variable="temperature",
            model="TEST-MODEL",
            scenario="test-scenario",
        )

        retrieved = manager.get(dataset_id)
        assert retrieved is not None
        assert "temperature" in retrieved.data_vars

    def test_store_extracts_metadata(self, manager, sample_dataset):
        """Test that store extracts metadata correctly."""
        dataset_id = manager.store(
            data=sample_dataset,
            source="test",
            variable="temperature",
        )

        metadata = manager.get_metadata(dataset_id)
        assert metadata.variable == "temperature"
        assert metadata.source == "test"
        assert metadata.time_range is not None
        assert metadata.spatial_bounds is not None

    def test_get_nonexistent_raises(self, manager):
        """Test that getting nonexistent dataset raises KeyError."""
        with pytest.raises(KeyError):
            manager.get("nonexistent_id")

    def test_list_datasets(self, manager, sample_dataset):
        """Test listing all datasets."""
        # Store multiple datasets
        manager.store(sample_dataset, source="test1", variable="var1")
        manager.store(sample_dataset, source="test2", variable="var2")

        datasets = manager.list_datasets()
        assert len(datasets) == 2

    def test_delete_dataset(self, manager, sample_dataset):
        """Test deleting a dataset."""
        dataset_id = manager.store(sample_dataset, source="test", variable="var")

        assert manager.delete(dataset_id) is True
        assert manager.delete(dataset_id) is False  # Already deleted

        with pytest.raises(KeyError):
            manager.get(dataset_id)

    def test_clear(self, manager, sample_dataset):
        """Test clearing all datasets."""
        manager.store(sample_dataset, source="test1", variable="var1")
        manager.store(sample_dataset, source="test2", variable="var2")

        assert len(manager.list_datasets()) == 2

        manager.clear()

        assert len(manager.list_datasets()) == 0


class TestDatasetInfo:
    """Test DatasetInfo dataclass."""

    def test_dataset_info_creation(self):
        """Test creating DatasetInfo."""
        info = DatasetInfo(
            dataset_id="ds_test",
            source="test-source",
            variable="tas",
            model="TEST-MODEL",
            scenario="ssp585",
        )

        assert info.dataset_id == "ds_test"
        assert info.source == "test-source"
        assert info.variable == "tas"
        assert info.model == "TEST-MODEL"
        assert info.scenario == "ssp585"


class TestGlobalSessionManager:
    """Test the global session manager instance."""

    def test_global_manager_exists(self):
        """Test that global session manager is available."""
        assert session_manager is not None
        assert isinstance(session_manager, SessionManager)
