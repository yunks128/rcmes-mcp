"""Tests for MCP tools."""

import numpy as np
import pytest
import xarray as xr

from rcmes_mcp.utils.session import session_manager


@pytest.fixture(autouse=True)
def clear_session():
    """Clear session before and after each test."""
    session_manager.clear()
    yield
    session_manager.clear()


@pytest.fixture
def sample_temperature_dataset():
    """Create a sample temperature dataset."""
    times = np.arange("2020-01-01", "2020-12-31", dtype="datetime64[D]")
    lats = np.linspace(32, 42, 20)  # California-ish
    lons = np.linspace(-124, -114, 20)

    # Generate fake temperature data (K)
    data = 280 + 20 * np.random.rand(len(times), len(lats), len(lons))

    ds = xr.Dataset(
        {
            "tasmax": (["time", "lat", "lon"], data, {"units": "K", "long_name": "Daily Maximum Temperature"}),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )
    return ds


@pytest.fixture
def stored_dataset(sample_temperature_dataset):
    """Store a dataset and return its ID."""
    dataset_id = session_manager.store(
        data=sample_temperature_dataset,
        source="test",
        variable="tasmax",
        model="TEST-MODEL",
        scenario="test",
    )
    return dataset_id


class TestDataAccessTools:
    """Test data access tools."""

    def test_list_available_models_import(self):
        """Test that list_available_models can be imported."""
        from rcmes_mcp.tools.data_access import list_available_models

        result = list_available_models()
        assert "models" in result
        assert len(result["models"]) > 0

    def test_list_available_variables_import(self):
        """Test that list_available_variables can be imported."""
        from rcmes_mcp.tools.data_access import list_available_variables

        result = list_available_variables()
        assert "variables" in result
        assert len(result["variables"]) > 0

    def test_list_available_scenarios_import(self):
        """Test that list_available_scenarios can be imported."""
        from rcmes_mcp.tools.data_access import list_available_scenarios

        result = list_available_scenarios()
        assert "scenarios" in result

    def test_list_loaded_datasets(self, stored_dataset):
        """Test listing loaded datasets."""
        from rcmes_mcp.tools.data_access import list_loaded_datasets

        result = list_loaded_datasets()
        assert result["dataset_count"] == 1
        assert len(result["datasets"]) == 1

    def test_get_dataset_info(self, stored_dataset):
        """Test getting dataset info."""
        from rcmes_mcp.tools.data_access import get_dataset_info

        result = get_dataset_info(stored_dataset)
        assert result["dataset_id"] == stored_dataset
        assert result["variable"] == "tasmax"

    def test_delete_dataset(self, stored_dataset):
        """Test deleting a dataset."""
        from rcmes_mcp.tools.data_access import delete_dataset, list_loaded_datasets

        result = delete_dataset(stored_dataset)
        assert result["success"] is True

        datasets = list_loaded_datasets()
        assert datasets["dataset_count"] == 0


class TestProcessingTools:
    """Test data processing tools."""

    def test_convert_units_kelvin_to_celsius(self, stored_dataset):
        """Test converting temperature from K to C."""
        from rcmes_mcp.tools.processing import convert_units

        result = convert_units(stored_dataset, "degC")
        assert result["success"] is True
        assert result["new_unit"] == "degC"

        # Verify the data was converted
        new_ds = session_manager.get(result["dataset_id"])
        # Original was ~280-300K, should now be ~7-27C
        assert float(new_ds["tasmax"].mean()) < 50  # Should be Celsius now

    def test_temporal_resample(self, stored_dataset):
        """Test resampling to monthly."""
        from rcmes_mcp.tools.processing import temporal_resample

        result = temporal_resample(stored_dataset, "monthly", "mean")
        assert result["success"] is True
        assert result["frequency"] == "monthly"

        new_ds = session_manager.get(result["dataset_id"])
        # Original had ~364 days, monthly should have ~12
        assert new_ds.dims["time"] <= 12

    def test_spatial_subset(self, stored_dataset):
        """Test spatial subsetting."""
        from rcmes_mcp.tools.processing import spatial_subset

        result = spatial_subset(stored_dataset, 35, 38, -120, -116)
        assert result["success"] is True

        new_ds = session_manager.get(result["dataset_id"])
        assert float(new_ds.lat.min()) >= 35
        assert float(new_ds.lat.max()) <= 38


class TestAnalysisTools:
    """Test analysis tools."""

    def test_calculate_statistics(self, stored_dataset):
        """Test calculating statistics."""
        from rcmes_mcp.tools.analysis import calculate_statistics

        result = calculate_statistics(stored_dataset)
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result

    def test_calculate_climatology(self, stored_dataset):
        """Test calculating monthly climatology."""
        from rcmes_mcp.tools.analysis import calculate_climatology

        result = calculate_climatology(stored_dataset, "monthly")
        assert result["success"] is True
        assert result["period"] == "monthly"

    def test_calculate_trend(self, stored_dataset):
        """Test calculating trend."""
        from rcmes_mcp.tools.analysis import calculate_trend

        result = calculate_trend(stored_dataset)
        assert "trend" in result
        assert "slope_per_decade" in result["trend"]
        assert "p_value" in result["trend"]

    def test_calculate_regional_mean(self, stored_dataset):
        """Test calculating regional mean."""
        from rcmes_mcp.tools.analysis import calculate_regional_mean

        result = calculate_regional_mean(stored_dataset, area_weighted=True)
        assert result["success"] is True
        assert "summary" in result


class TestIndicesTools:
    """Test climate indices tools."""

    def test_list_climate_indices(self):
        """Test listing available climate indices."""
        from rcmes_mcp.tools.indices import list_climate_indices

        result = list_climate_indices()
        assert "temperature_indices" in result
        assert "precipitation_indices" in result
        assert "TXx" in result["temperature_indices"]
        assert "Rx1day" in result["precipitation_indices"]


class TestCountryMasking:
    """Test country masking tools."""

    def test_list_countries(self):
        """Test listing available countries."""
        from rcmes_mcp.tools.processing import list_countries

        result = list_countries()
        assert "countries" in result
        assert len(result["countries"]) > 0
        assert result["count"] > 100  # Should have 100+ countries

    def test_mask_by_country(self, stored_dataset):
        """Test masking dataset to a country boundary."""
        from rcmes_mcp.tools.processing import mask_by_country

        # California-ish data should overlap with US
        result = mask_by_country(stored_dataset, "United States of America")
        assert result["success"] is True
        assert "dataset_id" in result
        assert result["country"] == "United States of America"

        # Verify masked data has some NaN (edges outside US)
        masked_ds = session_manager.get(result["dataset_id"])
        assert np.isnan(masked_ds["tasmax"].values).any()

    def test_mask_by_country_not_found(self, stored_dataset):
        """Test masking with nonexistent country."""
        from rcmes_mcp.tools.processing import mask_by_country

        result = mask_by_country(stored_dataset, "Atlantis")
        assert "error" in result

    def test_mask_by_country_case_insensitive(self, stored_dataset):
        """Test that country matching is case-insensitive."""
        from rcmes_mcp.tools.processing import mask_by_country

        result = mask_by_country(stored_dataset, "united states of america")
        assert result["success"] is True


class TestBatchETCCDI:
    """Test batch ETCCDI calculation."""

    def test_calculate_batch_etccdi_import(self):
        """Test that calculate_batch_etccdi can be imported."""
        from rcmes_mcp.tools.indices import calculate_batch_etccdi

        assert callable(calculate_batch_etccdi)

    def test_calculate_batch_etccdi_empty(self, stored_dataset):
        """Test batch ETCCDI with empty indices list."""
        from rcmes_mcp.tools.indices import calculate_batch_etccdi

        result = calculate_batch_etccdi(stored_dataset, [])
        assert "error" in result


class TestTrendFrequency:
    """Test calculate_trend frequency detection."""

    def test_trend_annual_data(self):
        """Test that trend correctly detects annual data."""
        from rcmes_mcp.tools.analysis import calculate_trend

        # Create annual time series data
        years = np.arange("2020", "2050", dtype="datetime64[Y]")
        lats = np.linspace(35, 40, 5)
        lons = np.linspace(-120, -115, 5)

        # Linear trend: 0.1 units per year = 1.0 units per decade
        trend_per_year = 0.1
        data = np.zeros((len(years), len(lats), len(lons)))
        for i in range(len(years)):
            data[i, :, :] = 280 + trend_per_year * i

        ds = xr.Dataset(
            {
                "tas": (["time", "lat", "lon"], data, {"units": "K"}),
            },
            coords={
                "time": years.astype("datetime64[ns]"),
                "lat": lats,
                "lon": lons,
            },
        )

        dataset_id = session_manager.store(
            data=ds, source="test", variable="tas", model="TEST", scenario="test",
        )

        result = calculate_trend(dataset_id)
        assert "trend" in result
        # For annual data with 0.1/year trend, slope_per_decade should be ~1.0
        assert abs(result["trend"]["slope_per_decade"] - 1.0) < 0.01

    def test_trend_daily_data(self, stored_dataset):
        """Test that trend works with daily data."""
        from rcmes_mcp.tools.analysis import calculate_trend

        result = calculate_trend(stored_dataset)
        assert "trend" in result
        assert "slope_per_decade" in result["trend"]


class TestVisualizationTools:
    """Test visualization tools."""

    def test_generate_histogram(self, stored_dataset):
        """Test generating a histogram."""
        from rcmes_mcp.tools.visualization import generate_histogram

        result = generate_histogram(stored_dataset, bins=20)
        assert result["success"] is True
        assert "image_base64" in result
        assert "statistics" in result
