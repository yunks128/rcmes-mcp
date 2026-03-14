"""
Cloud Access Utilities

Provides utilities for accessing climate data from cloud storage (S3, Azure Blob)
and remote services (OPeNDAP, THREDDS).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import tempfile
from pathlib import Path

import fsspec
import s3fs
import xarray as xr

# Local file cache directory for downloaded S3 files
_FILE_CACHE_DIR = Path(os.environ.get(
    "RCMES_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "rcmes_mcp_cache"),
)) / "s3_files"

# NEX-GDDP-CMIP6 dataset configuration
NEX_GDDP_CMIP6_BUCKET = "nex-gddp-cmip6"
NEX_GDDP_CMIP6_REGION = "us-west-2"

# THREDDS endpoints
NCCS_THREDDS_BASE = "https://ds.nccs.nasa.gov/thredds"

# Available models in NEX-GDDP-CMIP6 with descriptions
CMIP6_MODEL_INFO = {
    "ACCESS-CM2": "CSIRO-ARCCSS, Australia — coupled climate model",
    "ACCESS-ESM1-5": "CSIRO, Australia — earth system model with biogeochemistry",
    "BCC-CSM2-MR": "Beijing Climate Center, China — medium-resolution coupled model",
    "CanESM5": "CCCma, Canada — Canadian Earth System Model",
    "CESM2": "NCAR, USA — Community Earth System Model",
    "CESM2-WACCM": "NCAR, USA — CESM2 with whole atmosphere community climate model",
    "CMCC-CM2-SR5": "CMCC, Italy — coupled climate model",
    "CMCC-ESM2": "CMCC, Italy — earth system model with carbon cycle",
    "CNRM-CM6-1": "CNRM/Météo-France, France — coupled climate model",
    "CNRM-ESM2-1": "CNRM/Météo-France, France — earth system model",
    "EC-Earth3": "EC-Earth Consortium, Europe — multi-institutional climate model",
    "EC-Earth3-Veg-LR": "EC-Earth Consortium, Europe — with dynamic vegetation, low resolution",
    "FGOALS-g3": "CAS/IAP, China — Flexible Global Ocean-Atmosphere-Land System",
    "GFDL-CM4": "NOAA/GFDL, USA — coupled climate model",
    "GFDL-ESM4": "NOAA/GFDL, USA — earth system model with atmospheric chemistry",
    "GISS-E2-1-G": "NASA/GISS, USA — Goddard Institute for Space Studies model",
    "HadGEM3-GC31-LL": "Met Office, UK — Hadley Centre model, low resolution",
    "HadGEM3-GC31-MM": "Met Office, UK — Hadley Centre model, medium resolution",
    "IITM-ESM": "IITM, India — Indian Institute of Tropical Meteorology ESM",
    "INM-CM4-8": "INM, Russia — Institute for Numerical Mathematics model",
    "INM-CM5-0": "INM, Russia — Institute for Numerical Mathematics model v5",
    "IPSL-CM6A-LR": "IPSL, France — Institut Pierre-Simon Laplace model",
    "KACE-1-0-G": "NIMS/KMA, South Korea — Korean coupled climate model",
    "KIOST-ESM": "KIOST, South Korea — Korea Institute of Ocean Science & Technology ESM",
    "MIROC6": "JAMSTEC/U. Tokyo, Japan — coupled climate model",
    "MIROC-ES2L": "JAMSTEC/U. Tokyo, Japan — earth system model, low resolution",
    "MPI-ESM1-2-HR": "MPI-M, Germany — Max Planck Institute model, high resolution",
    "MPI-ESM1-2-LR": "MPI-M, Germany — Max Planck Institute model, low resolution",
    "MRI-ESM2-0": "MRI/JMA, Japan — Meteorological Research Institute ESM",
    "NESM3": "NUIST, China — Nanjing University earth system model",
    "NorESM2-LM": "NCC, Norway — Norwegian Earth System Model, low-medium res",
    "NorESM2-MM": "NCC, Norway — Norwegian Earth System Model, medium res",
    "TaiESM1": "RCEC/AS, Taiwan — Taiwan Earth System Model",
    "UKESM1-0-LL": "Met Office/NERC, UK — UK Earth System Model",
}

# List of model names for backward compatibility
CMIP6_MODELS = list(CMIP6_MODEL_INFO.keys())

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
    """Get an S3 filesystem for accessing NEX-GDDP-CMIP6 data.

    Configured with optimized settings for climate data access:
    - 8MB default block size for efficient large reads
    - Connection pooling for reduced latency
    - Read-ahead caching for sequential access
    """
    return s3fs.S3FileSystem(
        anon=True,
        default_block_size=8 * 1024 * 1024,  # 8MB blocks for better throughput
        default_cache_type="readahead",  # Pre-fetch data for sequential reads
        config_kwargs={
            "max_pool_connections": 25,  # Increase connection pool
            "connect_timeout": 30,
            "read_timeout": 60,
        },
    )


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
    chunks: dict[str, int] | None = {},
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    progress_callback: Any | None = None,
) -> xr.Dataset:
    """
    Open NEX-GDDP-CMIP6 dataset from S3.

    Uses xarray with Dask for lazy loading - data is only read when needed.
    Optimized for fast access with early spatial subsetting.

    Args:
        variable: Climate variable
        model: Climate model
        scenario: Emissions scenario
        start_year: Optional start year filter
        end_year: Optional end year filter
        chunks: Dask chunk sizes (default: optimized for typical regional queries)
        lat_bounds: Optional (min, max) latitude bounds for early subsetting
        lon_bounds: Optional (min, max) longitude bounds for early subsetting

    Returns:
        xarray Dataset with lazy-loaded data
    """
    ref_url = f"s3://carbonplan-share/nasa-nex-reference/references_prod/{model}_{scenario}/reference.parquet"

    storage_options = {
        'anon': True,
        'remote_protocol': 's3',
        'remote_options': {'anon': True, 'asynchronous': True},
        'target_protocol': 's3',
        'target_options': {'anon': True}
    }

    ds = xr.open_dataset(
        ref_url,
        engine="kerchunk",
        storage_options=storage_options,
        chunks=chunks  
    )

    # 1. Variable selection (Lazy)
    if variable in ds.data_vars:
        ds = ds[[variable]]

    # 2. Temporal slice (Lazy)
    if start_year or end_year:
        time_slice = slice(
            str(start_year) if start_year else None,
            str(end_year) if end_year else None
        )
        ds = ds.sel(time=time_slice)

    # 3. Spatial subset (Lazy)
    # This is critical when chunks=None. It ensures that when you 
    # eventually do call .load(), you only pull the regional bytes.
    ds = _apply_spatial_subset(ds, lat_bounds, lon_bounds)

    return ds


def _apply_spatial_subset(
    ds: xr.Dataset,
    lat_bounds: tuple[float, float] | None,
    lon_bounds: tuple[float, float] | None,
) -> xr.Dataset:
    """Apply spatial subsetting to dataset."""
    if lat_bounds or lon_bounds:
        sel_kwargs = {}
        if lat_bounds:
            sel_kwargs["lat"] = slice(lat_bounds[0], lat_bounds[1])
        if lon_bounds:
            sel_kwargs["lon"] = slice(lon_bounds[0], lon_bounds[1])
        if sel_kwargs:
            ds = ds.sel(**sel_kwargs)
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
