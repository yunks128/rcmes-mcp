"""
Cloud Access Utilities

Provides utilities for accessing climate data from cloud storage (S3, Azure Blob)
and remote services (OPeNDAP, THREDDS).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import fsspec
import s3fs
import xarray as xr

# NEX-GDDP-CMIP6 dataset configuration
NEX_GDDP_CMIP6_BUCKET = "nex-gddp-cmip6"
NEX_GDDP_CMIP6_REGION = "us-west-2"

# THREDDS endpoints
NCCS_THREDDS_BASE = "https://ds.nccs.nasa.gov/thredds"

# Available models in NEX-GDDP-CMIP6
CMIP6_MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "CESM2",
    "CESM2-WACCM",
    "CMCC-CM2-SR5",
    "CMCC-ESM2",
    "CNRM-CM6-1",
    "CNRM-ESM2-1",
    "EC-Earth3",
    "EC-Earth3-Veg-LR",
    "FGOALS-g3",
    "GFDL-CM4",
    "GFDL-ESM4",
    "GISS-E2-1-G",
    "HadGEM3-GC31-LL",
    "HadGEM3-GC31-MM",
    "IITM-ESM",
    "INM-CM4-8",
    "INM-CM5-0",
    "IPSL-CM6A-LR",
    "KACE-1-0-G",
    "KIOST-ESM",
    "MIROC6",
    "MIROC-ES2L",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NESM3",
    "NorESM2-LM",
    "NorESM2-MM",
    "TaiESM1",
    "UKESM1-0-LL",
]

# Available scenarios
CMIP6_SCENARIOS = {
    "historical": "Historical simulations (1950-2014)",
    "ssp126": "SSP1-2.6: Sustainability - Low emissions",
    "ssp245": "SSP2-4.5: Middle of the Road",
    "ssp370": "SSP3-7.0: Regional Rivalry - High emissions",
    "ssp585": "SSP5-8.5: Fossil-fueled Development - Very high emissions",
}

# Available variables
CMIP6_VARIABLES = {
    "hurs": {"long_name": "Near-Surface Relative Humidity", "units": "%"},
    "huss": {"long_name": "Near-Surface Specific Humidity", "units": "kg kg-1"},
    "pr": {"long_name": "Precipitation", "units": "kg m-2 s-1"},
    "rlds": {"long_name": "Surface Downwelling Longwave Radiation", "units": "W m-2"},
    "rsds": {"long_name": "Surface Downwelling Shortwave Radiation", "units": "W m-2"},
    "sfcWind": {"long_name": "Near-Surface Wind Speed", "units": "m s-1"},
    "tas": {"long_name": "Near-Surface Air Temperature", "units": "K"},
    "tasmax": {"long_name": "Daily Maximum Near-Surface Air Temperature", "units": "K"},
    "tasmin": {"long_name": "Daily Minimum Near-Surface Air Temperature", "units": "K"},
}


@lru_cache(maxsize=1)
def get_s3_filesystem() -> s3fs.S3FileSystem:
    """Get an S3 filesystem for accessing NEX-GDDP-CMIP6 data."""
    return s3fs.S3FileSystem(anon=True)


def build_nex_gddp_path(
    variable: str,
    model: str,
    scenario: str,
    year: int | None = None,
) -> str:
    """
    Build the S3 path for NEX-GDDP-CMIP6 data.

    Path structure: s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/{model}/{scenario}/r1i1p1f1/{variable}/

    Args:
        variable: Climate variable (tas, pr, etc.)
        model: Climate model name
        scenario: Emissions scenario (historical, ssp126, etc.)
        year: Optional specific year

    Returns:
        S3 path to the data
    """
    base_path = f"s3://{NEX_GDDP_CMIP6_BUCKET}/NEX-GDDP-CMIP6/{model}/{scenario}/r1i1p1f1/{variable}"

    if year:
        return f"{base_path}/{variable}_day_{model}_{scenario}_r1i1p1f1_gn_{year}.nc"

    return base_path


def list_available_files(
    model: str,
    scenario: str,
    variable: str,
) -> list[str]:
    """List available NetCDF files for a model/scenario/variable combination."""
    fs = get_s3_filesystem()
    path = build_nex_gddp_path(variable, model, scenario)

    try:
        files = fs.glob(f"{path}/*.nc")
        return sorted(files)
    except Exception:
        return []


def open_nex_gddp_dataset(
    variable: str,
    model: str,
    scenario: str,
    start_year: int | None = None,
    end_year: int | None = None,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """
    Open NEX-GDDP-CMIP6 dataset from S3.

    Uses xarray with Dask for lazy loading - data is only read when needed.

    Args:
        variable: Climate variable
        model: Climate model
        scenario: Emissions scenario
        start_year: Optional start year filter
        end_year: Optional end year filter
        chunks: Dask chunk sizes (default: auto-chunking)

    Returns:
        xarray Dataset with lazy-loaded data
    """
    if chunks is None:
        chunks = {"time": 365, "lat": 100, "lon": 100}

    fs = get_s3_filesystem()
    base_path = build_nex_gddp_path(variable, model, scenario)

    # Get list of files
    files = fs.glob(f"{base_path}/*.nc")

    if not files:
        raise FileNotFoundError(
            f"No data found for {model}/{scenario}/{variable}. "
            f"Check that the model and scenario are available."
        )

    # Filter by year if specified
    if start_year or end_year:
        filtered_files = []
        for f in files:
            # Extract year from filename
            try:
                year = int(f.split("_")[-1].replace(".nc", ""))
                if start_year and year < start_year:
                    continue
                if end_year and year > end_year:
                    continue
                filtered_files.append(f)
            except ValueError:
                continue
        files = filtered_files

    if not files:
        raise FileNotFoundError(
            f"No data found for {model}/{scenario}/{variable} "
            f"in year range {start_year}-{end_year}"
        )

    # Open multi-file dataset
    file_urls = [f"s3://{f}" for f in files]

    ds = xr.open_mfdataset(
        file_urls,
        engine="h5netcdf",
        chunks=chunks,
        combine="by_coords",
        parallel=True,
        storage_options={"anon": True},
    )

    return ds


def get_opendap_url(
    variable: str,
    model: str,
    scenario: str,
    year: int,
) -> str:
    """
    Get OPeNDAP URL for NEX-GDDP-CMIP6 data via NCCS THREDDS.

    Alternative access method when S3 direct access is not available.
    """
    filename = f"{variable}_day_{model}_{scenario}_r1i1p1f1_gn_{year}.nc"
    return f"{NCCS_THREDDS_BASE}/dodsC/AMES/NEX/GDDP-CMIP6/{model}/{scenario}/r1i1p1f1/{variable}/{filename}"


def validate_model(model: str) -> bool:
    """Check if model is available in NEX-GDDP-CMIP6."""
    return model in CMIP6_MODELS


def validate_scenario(scenario: str) -> bool:
    """Check if scenario is available in NEX-GDDP-CMIP6."""
    return scenario in CMIP6_SCENARIOS


def validate_variable(variable: str) -> bool:
    """Check if variable is available in NEX-GDDP-CMIP6."""
    return variable in CMIP6_VARIABLES
