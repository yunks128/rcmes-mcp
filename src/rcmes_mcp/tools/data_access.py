"""
Data Access Tools

MCP tools for loading and accessing climate data from various sources
including NEX-GDDP-CMIP6 on AWS S3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger("rcmes.tools.data_access")

from rcmes_mcp.server import mcp
from rcmes_mcp.utils.cache import result_cache
from rcmes_mcp.utils.cloud import (
    CMIP6_MODEL_INFO,
    CMIP6_MODELS,
    CMIP6_SCENARIOS,
    CMIP6_VARIABLES,
    list_available_files,
    open_nex_gddp_dataset,
    validate_model,
    validate_scenario,
    validate_variable,
)
from rcmes_mcp.utils.session import session_manager
from rcmes_mcp.utils.validation import validate_date_range, validate_lat_lon_bounds

# Thread-local storage for progress callback (avoids polluting MCP tool signatures)
_thread_local = threading.local()

# Encoding for compressed NetCDF cache files (~60-70% smaller)
_NC_ENCODING_DEFAULTS = {
    "zlib": True,
    "complevel": 4,  # balance between speed and ratio
}


def _get_subset_cache_key(
    variable: str,
    model: str,
    scenario: str,
    start_date: str,
    end_date: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> str:
    """Generate a cache key for a data subset request."""
    key_str = f"{variable}_{model}_{scenario}_{start_date}_{end_date}_{lat_min}_{lat_max}_{lon_min}_{lon_max}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _get_cached_subset(cache_key: str) -> xr.Dataset | None:
    """Try to retrieve a cached subset from disk."""
    cache_dir = result_cache.cache_dir / "subsets"
    cache_file = cache_dir / f"{cache_key}.nc"

    if cache_file.exists():
        try:
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < 24 * 3600:  # 24 hours
                return xr.open_dataset(cache_file, chunks="auto")
        except Exception:
            pass
    return None


def _get_cached_stats(cache_key: str) -> dict | None:
    """Retrieve pre-computed statistics for a cached subset."""
    stats_file = result_cache.cache_dir / "subsets" / f"{cache_key}_stats.json"
    if stats_file.exists():
        try:
            file_age = time.time() - stats_file.stat().st_mtime
            if file_age < 24 * 3600:
                with open(stats_file) as f:
                    return json.load(f)
        except Exception:
            pass
    return None


def _cache_subset(cache_key: str, ds: xr.Dataset) -> None:
    """Cache a subset to disk with compression, and pre-compute basic statistics."""
    try:
        cache_dir = result_cache.cache_dir / "subsets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.nc"

        # Only cache small-to-medium subsets (< 100MB estimated uncompressed)
        estimated_size = 1
        for dim_size in ds.dims.values():
            estimated_size *= dim_size
        estimated_size *= 6  # 4 bytes per float32 + overhead

        if estimated_size < 100 * 1024 * 1024:  # 100MB limit
            # Build per-variable encoding with zlib compression
            encoding = {}
            for var in ds.data_vars:
                encoding[var] = _NC_ENCODING_DEFAULTS.copy()
            ds.to_netcdf(cache_file, encoding=encoding)

            # Pre-compute and cache basic statistics alongside the data
            _precompute_stats(cache_key, ds)
    except Exception:
        pass  # Caching failures shouldn't break the main flow


# Track pending background materializations: dataset_id -> threading.Event
_materialization_events: dict[str, threading.Event] = {}


def _background_cache(cache_key: str, ds: xr.Dataset, dataset_id: str | None = None) -> None:
    """Materialize, cache to disk, and update session with in-memory data.

    After this completes, the session holds a fully materialized (in-memory)
    dataset, so subsequent tool calls (stats, viz, download) are instant.
    """
    done_event = threading.Event()
    if dataset_id:
        _materialization_events[dataset_id] = done_event

    def _do_cache():
        try:
            # Materialize the lazy dataset (this is the S3 download)
            materialized = ds.compute()
            logger.info(f"Background materialization complete for {cache_key}")

            # Swap the lazy dataset in the session with the materialized one
            if dataset_id and dataset_id in session_manager._datasets:
                session_manager._datasets[dataset_id] = materialized
                logger.info(f"Session updated with materialized data for {dataset_id}")

            # Cache to disk (from memory now, no second S3 hit)
            _cache_subset(cache_key, materialized)
            logger.info(f"Background disk cache complete for {cache_key}")
        except Exception as e:
            logger.warning(f"Background cache failed for {cache_key}: {e}")
        finally:
            done_event.set()
            _materialization_events.pop(dataset_id, None)

    t = threading.Thread(target=_do_cache, daemon=True)
    t.start()


def wait_for_materialization(dataset_id: str, timeout: float = 600) -> bool:
    """Wait for background materialization to finish. Returns True if done."""
    event = _materialization_events.get(dataset_id)
    if event is None:
        return True  # No pending materialization
    logger.info(f"Waiting for background materialization of {dataset_id}...")
    return event.wait(timeout=timeout)


def _precompute_stats(cache_key: str, ds: xr.Dataset) -> None:
    """Pre-compute basic statistics and save as JSON next to the cached subset."""
    try:
        stats_file = result_cache.cache_dir / "subsets" / f"{cache_key}_stats.json"
        import dask
        for var_name in ds.data_vars:
            data = ds[var_name]
            mean_val, std_val, min_val, max_val = dask.compute(
                data.mean(), data.std(), data.min(), data.max()
            )
            stats = {
                "variable": var_name,
                "mean": float(mean_val),
                "std": float(std_val),
                "min": float(min_val),
                "max": float(max_val),
            }
            with open(stats_file, "w") as f:
                json.dump(stats, f)
    except Exception:
        pass  # Pre-computation is best-effort


@mcp.tool()
def list_available_models(
    dataset: str = "NEX-GDDP-CMIP6",
    scenario: str | None = None,
) -> dict:
    """
    List available climate models in the dataset.

    Args:
        dataset: Dataset name. Currently supports "NEX-GDDP-CMIP6"
        scenario: Optional filter by emissions scenario (ssp126, ssp245, ssp370, ssp585)

    Returns:
        Dictionary with available models and their descriptions
    """
    if dataset != "NEX-GDDP-CMIP6":
        return {"error": f"Dataset '{dataset}' not yet supported. Use 'NEX-GDDP-CMIP6'."}

    models = CMIP6_MODELS.copy()

    # If scenario specified, could filter models that have data for that scenario
    # For now, all models have all scenarios

    models_with_info = [
        {"name": m, "description": CMIP6_MODEL_INFO.get(m, "")}
        for m in models
    ]

    return {
        "dataset": dataset,
        "model_count": len(models),
        "models": models,
        "models_info": models_with_info,
        "scenarios_available": list(CMIP6_SCENARIOS.keys()),
        "note": "All models are available for all scenarios (historical, ssp126, ssp245, ssp370, ssp585)",
    }


@mcp.tool()
def list_available_variables(dataset: str = "NEX-GDDP-CMIP6") -> dict:
    """
    List climate variables available in the dataset.

    Args:
        dataset: Dataset name. Currently supports "NEX-GDDP-CMIP6"

    Returns:
        Dictionary with variable names, descriptions, and units
    """
    if dataset != "NEX-GDDP-CMIP6":
        return {"error": f"Dataset '{dataset}' not yet supported. Use 'NEX-GDDP-CMIP6'."}

    variables = []
    for var_name, info in CMIP6_VARIABLES.items():
        variables.append({
            "name": var_name,
            "long_name": info["long_name"],
            "units": info["units"],
        })

    return {
        "dataset": dataset,
        "variable_count": len(variables),
        "variables": variables,
    }


@mcp.tool()
def list_available_scenarios(dataset: str = "NEX-GDDP-CMIP6") -> dict:
    """
    List available emissions scenarios in the dataset.

    Args:
        dataset: Dataset name

    Returns:
        Dictionary with scenario names and descriptions
    """
    if dataset != "NEX-GDDP-CMIP6":
        return {"error": f"Dataset '{dataset}' not yet supported."}

    scenarios = []
    for scenario_id, description in CMIP6_SCENARIOS.items():
        scenarios.append({
            "id": scenario_id,
            "description": description,
        })

    return {
        "dataset": dataset,
        "scenarios": scenarios,
        "note": "Use 'historical' for 1950-2014, SSP scenarios for 2015-2100",
    }


@mcp.tool()
def get_dataset_metadata(
    model: str,
    scenario: str,
    variable: str,
    dataset: str = "NEX-GDDP-CMIP6",
) -> dict:
    """
    Get detailed metadata for a specific model/scenario/variable combination.

    Args:
        model: Climate model name (e.g., "ACCESS-CM2", "CESM2")
        scenario: Emissions scenario (historical, ssp126, ssp245, ssp370, ssp585)
        variable: Climate variable (tas, tasmax, tasmin, pr, etc.)
        dataset: Dataset name

    Returns:
        Metadata including time range, spatial extent, file count
    """
    if dataset != "NEX-GDDP-CMIP6":
        return {"error": f"Dataset '{dataset}' not yet supported."}

    # Validate inputs
    if not validate_model(model):
        return {"error": f"Model '{model}' not found. Use list_available_models() to see available models."}

    if not validate_scenario(scenario):
        return {"error": f"Scenario '{scenario}' not found. Use list_available_scenarios() to see options."}

    if not validate_variable(variable):
        return {"error": f"Variable '{variable}' not found. Use list_available_variables() to see options."}

    # Get file list
    files = list_available_files(model, scenario, variable)

    if not files:
        return {
            "error": f"No data files found for {model}/{scenario}/{variable}",
            "suggestion": "This combination may not be available in the dataset.",
        }

    # Extract year range from filenames
    years = []
    for f in files:
        try:
            year = int(f.split("_")[-1].replace(".nc", ""))
            years.append(year)
        except ValueError:
            continue

    var_info = CMIP6_VARIABLES.get(variable, {})

    return {
        "model": model,
        "scenario": scenario,
        "variable": variable,
        "variable_long_name": var_info.get("long_name", ""),
        "units": var_info.get("units", ""),
        "file_count": len(files),
        "year_range": {
            "start": min(years) if years else None,
            "end": max(years) if years else None,
        },
        "spatial_resolution": "0.25° x 0.25° (~25 km)",
        "temporal_resolution": "daily",
        "spatial_coverage": {
            "lat_min": -60,
            "lat_max": 90,
            "lon_min": -180,
            "lon_max": 180,
        },
    }


@mcp.tool()
def load_climate_data(
    variable: str,
    model: str,
    scenario: str,
    start_date: str,
    end_date: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    dataset: str = "NEX-GDDP-CMIP6",
) -> dict:
    """
    Load climate data for a specific region and time period.

    This is the main data loading function. It returns a dataset ID that can be
    used in subsequent operations (subsetting, analysis, visualization).

    Args:
        variable: Climate variable (tas, tasmax, tasmin, pr, hurs, sfcWind, etc.)
        model: Climate model name (e.g., "ACCESS-CM2", "CESM2", "GFDL-ESM4")
        scenario: Emissions scenario (historical, ssp126, ssp245, ssp370, ssp585)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        lat_min: Minimum latitude (-60 to 90)
        lat_max: Maximum latitude (-60 to 90)
        lon_min: Minimum longitude (-180 to 180)
        lon_max: Maximum longitude (-180 to 180)
        dataset: Dataset name (default: NEX-GDDP-CMIP6)

    Returns:
        Dictionary with dataset_id for use in subsequent operations, plus metadata

    Example:
        # Load maximum temperature for California under high emissions
        load_climate_data(
            variable="tasmax",
            model="ACCESS-CM2",
            scenario="ssp585",
            start_date="2050-01-01",
            end_date="2050-12-31",
            lat_min=32.0,
            lat_max=42.0,
            lon_min=-124.0,
            lon_max=-114.0
        )
    """
    logger.info(
        "load_climate_data called",
        extra={"tool": "load_climate_data", "variable": variable,
               "model": model, "scenario": scenario},
    )
    t0 = time.perf_counter()

    if dataset != "NEX-GDDP-CMIP6":
        return {"error": f"Dataset '{dataset}' not yet supported."}

    # Validate inputs
    if not validate_model(model):
        return {"error": f"Model '{model}' not found.", "available_models": CMIP6_MODELS[:10]}

    if not validate_scenario(scenario):
        return {"error": f"Scenario '{scenario}' not found.", "available_scenarios": list(CMIP6_SCENARIOS.keys())}

    if not validate_variable(variable):
        return {"error": f"Variable '{variable}' not found.", "available_variables": list(CMIP6_VARIABLES.keys())}

    # Validate dates and coordinates
    try:
        start_date, end_date = validate_date_range(start_date, end_date)
    except ValueError as e:
        return {"error": str(e)}

    try:
        validate_lat_lon_bounds(lat_min, lat_max, lon_min, lon_max)
    except ValueError as e:
        return {"error": str(e)}

    # Extract years for S3 file selection
    start_year = int(start_date.split("-")[0])
    end_year = int(end_date.split("-")[0])

    # Convert longitude from -180/180 to 0-360 (dataset uses 0-360)
    lon_min_360 = lon_min if lon_min >= 0 else 360 + lon_min
    lon_max_360 = lon_max if lon_max >= 0 else 360 + lon_max

    # Handle wrap-around: if lon_min_360 >= lon_max_360 the range spans the
    # 0/360 boundary (e.g. global -180..180 → 180..180, or cross-meridian).
    # In that case skip longitude subsetting and select the full range.
    if lon_min_360 >= lon_max_360:
        lon_bounds_arg = None  # full longitude
    else:
        lon_bounds_arg = (lon_min_360, lon_max_360)

    # Progress callback helper
    progress_cb = getattr(_thread_local, 'progress_callback', None)
    def _progress(step: int, total: int, detail: str):
        if progress_cb:
            progress_cb(step, total, detail)

    TOTAL_STEPS = 5

    # Step 1: Check cache
    _progress(1, TOTAL_STEPS, "Checking local cache...")
    cache_key = _get_subset_cache_key(
        variable, model, scenario, start_date, end_date,
        lat_min, lat_max, lon_min_360, lon_max_360
    )
    cached_ds = _get_cached_subset(cache_key)

    try:
        if cached_ds is not None:
            _progress(2, TOTAL_STEPS, "Cache hit — loading from disk")
            ds = cached_ds
        else:
            # Step 2: Open dataset
            _progress(2, TOTAL_STEPS, f"Opening {model}/{scenario} from S3...")
            ds = open_nex_gddp_dataset(
                variable=variable,
                model=model,
                scenario=scenario,
                start_year=start_year,
                end_year=end_year,
                lat_bounds=(lat_min, lat_max),
                lon_bounds=lon_bounds_arg,
                progress_callback=progress_cb,
            )

            # Step 3: Subset by time
            _progress(3, TOTAL_STEPS, f"Subsetting {start_date} to {end_date}...")
            ds = ds.sel(time=slice(start_date, end_date))

        # Step 4: Store in session
        _progress(4, TOTAL_STEPS, "Storing in session...")
        dataset_id = session_manager.store(
            data=ds,
            source=dataset,
            variable=variable,
            model=model,
            scenario=scenario,
            description=f"{variable} from {model} ({scenario}) for {start_date} to {end_date}",
        )

        # Step 5: Start background materialization (non-blocking)
        if cached_ds is None:
            _progress(5, TOTAL_STEPS, "Materializing in background...")
            _background_cache(cache_key, ds, dataset_id=dataset_id)

        # Get shape info (triggers minimal data read)
        time_size = ds.dims.get("time", 0)
        lat_size = ds.dims.get("lat", 0)
        lon_size = ds.dims.get("lon", 0)

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            f"Loaded {variable}/{model}/{scenario} → {dataset_id} "
            f"({time_size}×{lat_size}×{lon_size}) in {duration_ms}ms",
            extra={"tool": "load_climate_data", "dataset_id": dataset_id,
                   "variable": variable, "model": model, "scenario": scenario,
                   "duration_ms": duration_ms},
        )

        result = {
            "success": True,
            "dataset_id": dataset_id,
            "variable": variable,
            "model": model,
            "scenario": scenario,
            "time_range": {"start": start_date, "end": end_date},
            "spatial_bounds": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            "dimensions": {
                "time": time_size,
                "lat": lat_size,
                "lon": lon_size,
            },
            "note": f"Use dataset_id '{dataset_id}' in subsequent operations.",
        }

        # Attach pre-computed statistics if available (instant, no compute)
        cached_stats = _get_cached_stats(cache_key)
        if cached_stats:
            result["precomputed_statistics"] = cached_stats

        return result

    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}", extra={"tool": "load_climate_data", "error": str(e)})
        return {"error": str(e)}
    except PermissionError as e:
        logger.error(f"S3 access error: {e}", extra={"tool": "load_climate_data", "error": str(e)})
        return {
            "error": f"S3 access denied — this usually means the data server is unreachable from this network. "
                     f"Try using pre-downloaded local data (RCMES_LOCAL_DATA_DIR) or run 'rcmes-warmup' from a machine with S3 access.",
        }
    except Exception as e:
        err_str = str(e)
        # Provide helpful context for common network errors
        if "Access Denied" in err_str or "PermissionError" in err_str:
            err_str = (
                f"S3 access denied — the data server may be unreachable from this network. "
                f"Try using pre-downloaded local data (RCMES_LOCAL_DATA_DIR) or run 'rcmes-warmup' from a machine with S3 access."
            )
        elif "timed out" in err_str.lower() or "timeout" in err_str.lower():
            err_str = (
                f"Connection to S3 timed out — the data server may be unreachable from this network. "
                f"Try using pre-downloaded local data (RCMES_LOCAL_DATA_DIR)."
            )
        logger.exception(f"Failed to load data: {e}", extra={"tool": "load_climate_data", "error": str(e)})
        return {"error": f"Failed to load data: {err_str}"}


@mcp.tool()
def list_loaded_datasets() -> dict:
    """
    List all datasets currently loaded in the session.

    Returns:
        Dictionary with list of loaded datasets and their metadata
    """
    datasets = session_manager.list_datasets()

    return {
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


@mcp.tool()
def get_dataset_info(dataset_id: str) -> dict:
    """
    Get detailed information about a loaded dataset.

    Args:
        dataset_id: ID of the loaded dataset

    Returns:
        Detailed metadata and statistics about the dataset
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get dimension info
    dims = dict(ds.dims)

    # Get variable info
    if isinstance(ds, xr.DataArray):
        var_info = {
            "name": ds.name,
            "dtype": str(ds.dtype),
            "units": ds.attrs.get("units", "unknown"),
        }
    else:
        var_info = {}
        for var in ds.data_vars:
            var_info[var] = {
                "dtype": str(ds[var].dtype),
                "units": ds[var].attrs.get("units", "unknown"),
            }

    return {
        "dataset_id": dataset_id,
        "source": metadata.source,
        "variable": metadata.variable,
        "model": metadata.model,
        "scenario": metadata.scenario,
        "dimensions": dims,
        "variables": var_info,
        "time_range": metadata.time_range,
        "spatial_bounds": metadata.spatial_bounds,
        "description": metadata.description,
    }


@mcp.tool()
def delete_dataset(dataset_id: str) -> dict:
    """
    Delete a dataset from the session to free memory.

    Args:
        dataset_id: ID of the dataset to delete

    Returns:
        Confirmation of deletion
    """
    success = session_manager.delete(dataset_id)

    if success:
        return {"success": True, "message": f"Dataset '{dataset_id}' deleted."}
    else:
        return {"error": f"Dataset '{dataset_id}' not found."}
