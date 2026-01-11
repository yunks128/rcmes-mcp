"""
Data Processing Tools

MCP tools for processing climate data including temporal/spatial
subsetting, regridding, and unit conversions.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from rcmes_mcp.server import mcp
from rcmes_mcp.utils.session import session_manager


@mcp.tool()
def temporal_subset(
    dataset_id: str,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Subset a dataset to a specific time period.

    Args:
        dataset_id: ID of the dataset to subset
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        New dataset_id for the subsetted data
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    try:
        ds_subset = ds.sel(time=slice(start_date, end_date))
    except Exception as e:
        return {"error": f"Failed to subset time: {str(e)}"}

    new_id = session_manager.store(
        data=ds_subset,
        source=metadata.source,
        variable=metadata.variable,
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"Temporal subset of {dataset_id}: {start_date} to {end_date}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "time_range": {"start": start_date, "end": end_date},
        "time_steps": ds_subset.dims.get("time", 0),
    }


@mcp.tool()
def spatial_subset(
    dataset_id: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> dict:
    """
    Subset a dataset to a geographic bounding box.

    Args:
        dataset_id: ID of the dataset to subset
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        lon_min: Minimum longitude
        lon_max: Maximum longitude

    Returns:
        New dataset_id for the subsetted data
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    try:
        ds_subset = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    except Exception as e:
        return {"error": f"Failed to subset spatially: {str(e)}"}

    new_id = session_manager.store(
        data=ds_subset,
        source=metadata.source,
        variable=metadata.variable,
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"Spatial subset of {dataset_id}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "spatial_bounds": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
    }


@mcp.tool()
def temporal_resample(
    dataset_id: str,
    frequency: str,
    method: str = "mean",
) -> dict:
    """
    Resample dataset to a different temporal frequency.

    Args:
        dataset_id: ID of the dataset to resample
        frequency: Target frequency - "monthly", "seasonal", "annual"
        method: Aggregation method - "mean" (default), "sum", "min", "max"
                Use "sum" for precipitation, "mean" for temperature

    Returns:
        New dataset_id for the resampled data
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Map frequency to pandas offset
    freq_map = {
        "monthly": "ME",  # Month end
        "seasonal": "QE-NOV",  # Seasonal (DJF, MAM, JJA, SON)
        "annual": "YE",  # Year end
    }

    if frequency not in freq_map:
        return {"error": f"Invalid frequency '{frequency}'. Use: monthly, seasonal, annual"}

    pandas_freq = freq_map[frequency]

    try:
        resampler = ds.resample(time=pandas_freq)

        if method == "mean":
            ds_resampled = resampler.mean()
        elif method == "sum":
            ds_resampled = resampler.sum()
        elif method == "min":
            ds_resampled = resampler.min()
        elif method == "max":
            ds_resampled = resampler.max()
        else:
            return {"error": f"Invalid method '{method}'. Use: mean, sum, min, max"}

    except Exception as e:
        return {"error": f"Failed to resample: {str(e)}"}

    new_id = session_manager.store(
        data=ds_resampled,
        source=metadata.source,
        variable=metadata.variable,
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"{frequency.capitalize()} {method} of {dataset_id}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "frequency": frequency,
        "method": method,
        "time_steps": ds_resampled.dims.get("time", 0),
    }


@mcp.tool()
def convert_units(
    dataset_id: str,
    target_unit: str,
) -> dict:
    """
    Convert dataset units to a different unit.

    Common conversions:
    - Temperature: K -> degC, K -> degF
    - Precipitation: kg m-2 s-1 -> mm/day, mm/month

    Args:
        dataset_id: ID of the dataset to convert
        target_unit: Target unit (degC, degF, mm/day, mm/month)

    Returns:
        New dataset_id with converted units
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    variable = metadata.variable

    # Get the data variable
    if isinstance(ds, xr.DataArray):
        data = ds
        current_unit = ds.attrs.get("units", "unknown")
    else:
        if variable in ds.data_vars:
            data = ds[variable]
            current_unit = data.attrs.get("units", "unknown")
        else:
            # Get first data variable
            var_name = list(ds.data_vars)[0]
            data = ds[var_name]
            current_unit = data.attrs.get("units", "unknown")
            variable = var_name

    # Temperature conversions
    if target_unit.lower() in ["degc", "celsius", "c"]:
        if current_unit == "K":
            data = data - 273.15
            data.attrs["units"] = "degC"
        elif current_unit in ["degC", "C"]:
            pass  # Already in Celsius
        else:
            return {"error": f"Cannot convert from {current_unit} to Celsius"}

    elif target_unit.lower() in ["degf", "fahrenheit", "f"]:
        if current_unit == "K":
            data = (data - 273.15) * 9 / 5 + 32
            data.attrs["units"] = "degF"
        elif current_unit in ["degC", "C"]:
            data = data * 9 / 5 + 32
            data.attrs["units"] = "degF"
        else:
            return {"error": f"Cannot convert from {current_unit} to Fahrenheit"}

    # Precipitation conversions (kg m-2 s-1 = mm/s)
    elif target_unit.lower() == "mm/day":
        if current_unit == "kg m-2 s-1":
            data = data * 86400  # seconds per day
            data.attrs["units"] = "mm/day"
        else:
            return {"error": f"Cannot convert from {current_unit} to mm/day"}

    elif target_unit.lower() == "mm/month":
        if current_unit == "kg m-2 s-1":
            data = data * 86400 * 30  # Approximate month
            data.attrs["units"] = "mm/month"
        else:
            return {"error": f"Cannot convert from {current_unit} to mm/month"}

    else:
        return {"error": f"Unsupported target unit: {target_unit}"}

    # Create new dataset
    if isinstance(ds, xr.DataArray):
        ds_converted = data
    else:
        ds_converted = ds.copy()
        ds_converted[variable] = data

    new_id = session_manager.store(
        data=ds_converted,
        source=metadata.source,
        variable=variable,
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"{dataset_id} converted to {target_unit}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "original_unit": current_unit,
        "new_unit": target_unit,
    }


@mcp.tool()
def regrid(
    dataset_id: str,
    target_resolution: float | None = None,
    target_dataset_id: str | None = None,
    method: str = "bilinear",
) -> dict:
    """
    Regrid dataset to a new resolution or to match another dataset's grid.

    Args:
        dataset_id: ID of the dataset to regrid
        target_resolution: Target resolution in degrees (e.g., 0.5 for 0.5° grid)
        target_dataset_id: Alternative - match the grid of another dataset
        method: Interpolation method - "bilinear", "nearest"

    Returns:
        New dataset_id for the regridded data
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    if target_resolution is None and target_dataset_id is None:
        return {"error": "Must specify either target_resolution or target_dataset_id"}

    try:
        if target_resolution:
            # Create new lat/lon grid
            lat_min = float(ds.lat.min())
            lat_max = float(ds.lat.max())
            lon_min = float(ds.lon.min())
            lon_max = float(ds.lon.max())

            new_lat = np.arange(lat_min, lat_max + target_resolution, target_resolution)
            new_lon = np.arange(lon_min, lon_max + target_resolution, target_resolution)

            ds_regridded = ds.interp(lat=new_lat, lon=new_lon, method=method)

        else:
            # Match another dataset's grid
            target_ds = session_manager.get(target_dataset_id)
            ds_regridded = ds.interp(lat=target_ds.lat, lon=target_ds.lon, method=method)

    except Exception as e:
        return {"error": f"Failed to regrid: {str(e)}"}

    new_id = session_manager.store(
        data=ds_regridded,
        source=metadata.source,
        variable=metadata.variable,
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"Regridded {dataset_id} to {target_resolution or 'target grid'}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "target_resolution": target_resolution,
        "method": method,
        "new_dimensions": {
            "lat": ds_regridded.dims.get("lat", 0),
            "lon": ds_regridded.dims.get("lon", 0),
        },
    }


@mcp.tool()
def calculate_anomaly(
    dataset_id: str,
    baseline_start: str,
    baseline_end: str,
) -> dict:
    """
    Calculate anomalies relative to a baseline period climatology.

    Anomaly = Value - Climatological Mean

    Args:
        dataset_id: ID of the dataset
        baseline_start: Start date of baseline period (YYYY-MM-DD)
        baseline_end: End date of baseline period (YYYY-MM-DD)

    Returns:
        New dataset_id containing anomaly values
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    try:
        # Extract baseline period
        baseline = ds.sel(time=slice(baseline_start, baseline_end))

        # Calculate climatological mean for each day of year
        climatology = baseline.groupby("time.dayofyear").mean()

        # Calculate anomaly
        anomaly = ds.groupby("time.dayofyear") - climatology

    except Exception as e:
        return {"error": f"Failed to calculate anomaly: {str(e)}"}

    new_id = session_manager.store(
        data=anomaly,
        source=metadata.source,
        variable=f"{metadata.variable}_anomaly",
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"Anomaly of {dataset_id} relative to {baseline_start}-{baseline_end}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "baseline_period": {
            "start": baseline_start,
            "end": baseline_end,
        },
        "description": "Values represent departure from baseline climatology",
    }
