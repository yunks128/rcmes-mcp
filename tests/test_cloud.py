"""Tests for cloud access utilities."""

import pytest

from rcmes_mcp.utils.cloud import (
    CMIP6_MODELS,
    CMIP6_SCENARIOS,
    CMIP6_VARIABLES,
    build_nex_gddp_path,
    get_s3_filesystem,
    list_available_files,
    validate_model,
    validate_scenario,
    validate_variable,
)


class TestCloudConstants:
    """Test cloud module constants."""

    def test_cmip6_models_not_empty(self):
        """Verify we have CMIP6 models defined."""
        assert len(CMIP6_MODELS) > 0
        assert "ACCESS-CM2" in CMIP6_MODELS
        assert "CESM2" in CMIP6_MODELS

    def test_cmip6_scenarios_complete(self):
        """Verify all SSP scenarios are defined."""
        expected = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
        for scenario in expected:
            assert scenario in CMIP6_SCENARIOS

    def test_cmip6_variables_have_metadata(self):
        """Verify variables have required metadata."""
        for var_name, info in CMIP6_VARIABLES.items():
            assert "long_name" in info
            assert "units" in info


class TestValidation:
    """Test validation functions."""

    def test_validate_model_valid(self):
        """Test validation of valid model names."""
        assert validate_model("ACCESS-CM2") is True
        assert validate_model("CESM2") is True
        assert validate_model("GFDL-ESM4") is True

    def test_validate_model_invalid(self):
        """Test validation of invalid model names."""
        assert validate_model("INVALID-MODEL") is False
        assert validate_model("") is False

    def test_validate_scenario_valid(self):
        """Test validation of valid scenarios."""
        assert validate_scenario("historical") is True
        assert validate_scenario("ssp585") is True

    def test_validate_scenario_invalid(self):
        """Test validation of invalid scenarios."""
        assert validate_scenario("rcp85") is False
        assert validate_scenario("future") is False

    def test_validate_variable_valid(self):
        """Test validation of valid variables."""
        assert validate_variable("tas") is True
        assert validate_variable("pr") is True
        assert validate_variable("tasmax") is True

    def test_validate_variable_invalid(self):
        """Test validation of invalid variables."""
        assert validate_variable("temperature") is False
        assert validate_variable("precip") is False


class TestPathBuilding:
    """Test S3 path construction."""

    def test_build_nex_gddp_path_base(self):
        """Test base path construction."""
        path = build_nex_gddp_path("tas", "ACCESS-CM2", "ssp585")
        assert "nex-gddp-cmip6" in path
        assert "ACCESS-CM2" in path
        assert "ssp585" in path
        assert "tas" in path

    def test_build_nex_gddp_path_with_year(self):
        """Test path construction with specific year."""
        path = build_nex_gddp_path("tasmax", "CESM2", "ssp245", year=2050)
        assert "2050.nc" in path
        assert "tasmax_day_CESM2_ssp245" in path


class TestS3Access:
    """Test S3 filesystem access (requires network)."""

    @pytest.fixture
    def s3_fs(self):
        """Get S3 filesystem."""
        return get_s3_filesystem()

    def test_get_s3_filesystem(self, s3_fs):
        """Test that we can create an S3 filesystem."""
        assert s3_fs is not None

    def test_list_available_files(self):
        """Test listing files from S3."""
        files = list_available_files("ACCESS-CM2", "ssp585", "tasmax")
        assert len(files) > 0
        # Files should be NetCDF
        assert all(f.endswith(".nc") for f in files)

    def test_list_available_files_invalid_model(self):
        """Test listing files for invalid model returns empty."""
        files = list_available_files("INVALID-MODEL", "ssp585", "tasmax")
        assert len(files) == 0
