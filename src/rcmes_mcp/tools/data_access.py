"""
Data Access Tools

MCP tools for loading and accessing climate data from various sources
including NEX-GDDP-CMIP6 on AWS S3.
"""

from __future__ import annotations

import xarray as xr

from rcmes_mcp.server import mcp
from rcmes_mcp.utils.cloud import (
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

    return {
        "dataset": dataset,
        "model_count": len(models),
        "models": models,
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
    if dataset != "NEX-GDDP-CMIP6":
        return {"error": f"Dataset '{dataset}' not yet supported."}

    # Validate inputs
    if not validate_model(model):
        return {"error": f"Model '{model}' not found.", "available_models": CMIP6_MODELS[:10]}

    if not validate_scenario(scenario):
        return {"error": f"Scenario '{scenario}' not found.", "available_scenarios": list(CMIP6_SCENARIOS.keys())}

    if not validate_variable(variable):
        return {"error": f"Variable '{variable}' not found.", "available_variables": list(CMIP6_VARIABLES.keys())}

    # Parse dates and extract years
    try:
        start_year = int(start_date.split("-")[0])
        end_year = int(end_date.split("-")[0])
    except (ValueError, IndexError):
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    # Validate coordinate ranges
    if not (-60 <= lat_min <= 90 and -60 <= lat_max <= 90):
        return {"error": "Latitude must be between -60 and 90."}
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
        return {"error": "Longitude must be between -180 and 180."}
    if lat_min >= lat_max:
        return {"error": "lat_min must be less than lat_max."}
    if lon_min >= lon_max:
        return {"error": "lon_min must be less than lon_max."}

    try:
        # Open dataset from S3
        ds = open_nex_gddp_dataset(
            variable=variable,
            model=model,
            scenario=scenario,
            start_year=start_year,
            end_year=end_year,
        )

        # Subset by time
        ds = ds.sel(time=slice(start_date, end_date))

        # Convert longitude from -180/180 to 0-360 if needed (dataset uses 0-360)
        lon_min_360 = lon_min if lon_min >= 0 else 360 + lon_min
        lon_max_360 = lon_max if lon_max >= 0 else 360 + lon_max

        # Subset by spatial bounds
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_360, lon_max_360))

        # Store in session
        dataset_id = session_manager.store(
            data=ds,
            source=dataset,
            variable=variable,
            model=model,
            scenario=scenario,
            description=f"{variable} from {model} ({scenario}) for {start_date} to {end_date}",
        )

        # Get shape info (triggers minimal data read)
        time_size = ds.dims.get("time", 0)
        lat_size = ds.dims.get("lat", 0)
        lon_size = ds.dims.get("lon", 0)

        return {
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

    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to load data: {str(e)}"}


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
