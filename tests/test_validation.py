"""Tests for the shared validation module."""

import numpy as np
import pytest
import xarray as xr

from rcmes_mcp.utils.validation import (
    check_download_size,
    validate_date,
    validate_date_range,
    validate_lat_lon_bounds,
)


# --- validate_date ---


class TestValidateDate:
    def test_valid_date(self):
        assert validate_date("2050-01-01") == "2050-01-01"

    def test_valid_date_normalizes(self):
        assert validate_date("2050-1-1") == "2050-01-01"

    def test_invalid_format_slash(self):
        with pytest.raises(ValueError, match="Invalid date"):
            validate_date("2050/01/01")

    def test_invalid_format_reversed(self):
        with pytest.raises(ValueError, match="Invalid date"):
            validate_date("01-01-2050")

    def test_invalid_not_a_date(self):
        with pytest.raises(ValueError, match="Invalid date"):
            validate_date("not-a-date")

    def test_invalid_none(self):
        with pytest.raises(ValueError, match="Invalid date"):
            validate_date(None)

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid date"):
            validate_date("")


# --- validate_date_range ---


class TestValidateDateRange:
    def test_valid_range(self):
        start, end = validate_date_range("2020-01-01", "2050-12-31")
        assert start == "2020-01-01"
        assert end == "2050-12-31"

    def test_start_after_end(self):
        with pytest.raises(ValueError, match="must be before"):
            validate_date_range("2050-01-01", "2020-01-01")

    def test_same_date(self):
        with pytest.raises(ValueError, match="must be before"):
            validate_date_range("2050-01-01", "2050-01-01")

    def test_span_too_long(self):
        with pytest.raises(ValueError, match="exceeding the maximum"):
            validate_date_range("1800-01-01", "2100-12-31")

    def test_custom_max_span(self):
        with pytest.raises(ValueError, match="exceeding the maximum"):
            validate_date_range("2020-01-01", "2025-01-01", max_span_days=365)

    def test_invalid_start_date(self):
        with pytest.raises(ValueError, match="Invalid date"):
            validate_date_range("bad-date", "2050-01-01")

    def test_invalid_end_date(self):
        with pytest.raises(ValueError, match="Invalid date"):
            validate_date_range("2020-01-01", "bad-date")


# --- validate_lat_lon_bounds ---


class TestValidateLatLonBounds:
    def test_valid_california(self):
        validate_lat_lon_bounds(32.0, 42.0, -124.0, -114.0)

    def test_valid_global(self):
        validate_lat_lon_bounds(-60.0, 90.0, -180.0, 179.9)

    def test_lat_below_range(self):
        with pytest.raises(ValueError, match="lat_min.*out of range"):
            validate_lat_lon_bounds(-70.0, 42.0, -124.0, -114.0)

    def test_lat_above_range(self):
        with pytest.raises(ValueError, match="lat_max.*out of range"):
            validate_lat_lon_bounds(32.0, 95.0, -124.0, -114.0)

    def test_lat_min_gte_max(self):
        with pytest.raises(ValueError, match="lat_min.*must be less than"):
            validate_lat_lon_bounds(42.0, 32.0, -124.0, -114.0)

    def test_lon_below_range(self):
        with pytest.raises(ValueError, match="lon_min.*out of range"):
            validate_lat_lon_bounds(32.0, 42.0, -200.0, -114.0)

    def test_lon_above_range(self):
        with pytest.raises(ValueError, match="lon_max.*out of range"):
            validate_lat_lon_bounds(32.0, 42.0, -124.0, 200.0)

    def test_lon_min_gte_max(self):
        with pytest.raises(ValueError, match="lon_min.*must be less than"):
            validate_lat_lon_bounds(32.0, 42.0, -114.0, -124.0)


# --- check_download_size ---


class TestCheckDownloadSize:
    def test_small_dataset_passes(self):
        ds = xr.Dataset({"temp": (["time", "lat", "lon"], np.zeros((10, 5, 5)))})
        check_download_size(ds)  # should not raise

    def test_small_dataarray_passes(self):
        da = xr.DataArray(np.zeros((10, 5, 5)), dims=["time", "lat", "lon"])
        check_download_size(da)  # should not raise

    def test_large_dataset_rejected(self):
        # Create a dataset with many elements using dask (doesn't allocate memory)
        import dask.array as dask_array
        data = dask_array.zeros((1000, 1000, 1000), chunks=(100, 100, 100))
        ds = xr.Dataset({"temp": (["time", "lat", "lon"], data)})
        with pytest.raises(ValueError, match="too large to download"):
            check_download_size(ds, max_elements=100_000_000)

    def test_custom_limit(self):
        ds = xr.Dataset({"temp": (["time", "lat", "lon"], np.zeros((10, 10, 10)))})
        with pytest.raises(ValueError, match="too large to download"):
            check_download_size(ds, max_elements=100)
