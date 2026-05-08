"""
MMGIS Integration Tools

MCP tools for exporting RCMES climate datasets to geospatial formats and
pushing them as live layers into a running MMGIS instance.

Workflow:
    load_climate_data(...)           → dataset_id
    calculate_statistics(dataset_id) → stats
    export_climate_geotiff(...)      → { file_path, cog_url }
    push_layer_to_mmgis(...)         → { browser_url }   ← MMGIS map opens
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger("rcmes.tools.mmgis")

from rcmes_mcp.server import mcp
from rcmes_mcp.utils.session import session_manager

# Shared data directory written by rcmes-mcp and served by TiTiler / MMGIS
def _ensure_data_dir() -> Path:
    """Create the data directory on first use (not at import time)."""
    data_dir = Path(os.environ.get("MMGIS_DATA_DIR", str(Path.home() / ".rcmes" / "layers")))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def _rcmes_file_url(filename: str) -> str:
    """Return the public URL for a file written to _DATA_DIR."""
    base = os.environ.get("RCMES_EXTERNAL_URL", "http://localhost:8502")
    return f"{base.rstrip('/')}/files/{filename}"


# ─── Tool 1: Export as Cloud-Optimized GeoTIFF ────────────────────────────


@mcp.tool()
def export_climate_geotiff(
    dataset_id: str,
    variable: str | None = None,
    time_aggregation: str = "mean",
    time_index: int | None = None,
    output_filename: str | None = None,
) -> dict:
    """
    Export a climate dataset variable to a Cloud-Optimized GeoTIFF (COG).

    The COG is written to the shared /data/layers directory where TiTiler can
    serve it as raster tiles for MMGIS. Use the returned cog_url with
    push_layer_to_mmgis to add it as a live map layer.

    Args:
        dataset_id: ID of the dataset in the session (from load_climate_data).
        variable: Variable name to export (e.g. "tas", "pr"). If None, uses
                  the first data variable in the dataset.
        time_aggregation: How to collapse the time dimension — "mean", "max",
                          "min", "std", or "none" (keep all time steps).
                          Ignored if time_index is set.
        time_index: Export a single time step by index (0-based). Takes
                    precedence over time_aggregation.
        output_filename: Override the output file name (must end in .tif).
                         Defaults to auto-generated name.

    Returns:
        dict with keys:
            file_path   — absolute path on disk
            cog_url     — URL for TiTiler to serve this COG (use in push_layer_to_mmgis)
            variable    — variable that was exported
            bbox        — [lon_min, lat_min, lon_max, lat_max]
            crs         — coordinate reference system string
            shape       — [height, width] of the raster
            time_label  — human-readable description of the time slice
    """
    import numpy as np

    try:
        import rioxarray  # noqa: F401  registers .rio accessor
    except ImportError as exc:
        raise RuntimeError(
            "rioxarray is required for GeoTIFF export. "
            "Install with: pip install rioxarray"
        ) from exc

    try:
        ds = session_manager.get(dataset_id)
    except KeyError:
        raise ValueError(f"Dataset '{dataset_id}' not found in session.")

    # Resolve variable
    if variable is None:
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError("Dataset has no data variables.")
        variable = data_vars[0]
    if variable not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not in dataset. Available: {list(ds.data_vars)}")

    da = ds[variable]

    # Resolve time dimension
    time_label: str
    if time_index is not None:
        if "time" not in da.dims:
            raise ValueError("Dataset has no 'time' dimension; cannot use time_index.")
        da = da.isel(time=time_index)
        t = ds.time.values[time_index]
        time_label = str(t)[:10]
    elif "time" in da.dims:
        agg_fn = {
            "mean": da.mean,
            "max": da.max,
            "min": da.min,
            "std": da.std,
        }
        if time_aggregation == "none":
            time_label = "all_times"
        elif time_aggregation in agg_fn:
            da = agg_fn[time_aggregation](dim="time")
            time_label = f"{time_aggregation}_all_times"
        else:
            raise ValueError(
                f"Unknown time_aggregation '{time_aggregation}'. "
                "Use 'mean', 'max', 'min', 'std', or 'none'."
            )
    else:
        time_label = "snapshot"

    # Ensure lat/lon are in the right order (lat descending for GeoTIFF north-up)
    lat_dim = next((d for d in da.dims if d in ("lat", "latitude")), None)
    lon_dim = next((d for d in da.dims if d in ("lon", "longitude")), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(
            f"Cannot find lat/lon dimensions in data array dims: {da.dims}"
        )

    # Transpose to (lat, lon) if needed
    if da.dims[-2:] != (lat_dim, lon_dim):
        da = da.transpose(..., lat_dim, lon_dim)

    # Ensure lat is descending (north-up) and lon is in -180/180 range
    if len(da[lat_dim]) > 1 and da[lat_dim].values[0] < da[lat_dim].values[-1]:
        da = da.isel({lat_dim: slice(None, None, -1)})
    lon_vals = da[lon_dim].values
    if lon_vals.max() > 180:
        da[lon_dim] = (da[lon_dim] + 180) % 360 - 180
        da = da.sortby(lon_dim)

    # Write CRS first, then set spatial dims so rioxarray can find them
    da = da.rio.write_crs("EPSG:4326", inplace=False)
    da = da.rio.set_spatial_dims(x_dim=lon_dim, y_dim=lat_dim)

    # Fill NaN with a nodata sentinel for GeoTIFF compatibility
    nodata = -9999.0
    da_filled = da.fillna(nodata).astype("float32")
    # Re-apply spatial dims and CRS on the filled array (fillna creates a new object)
    da_filled = da_filled.rio.write_crs("EPSG:4326", inplace=False)
    da_filled = da_filled.rio.set_spatial_dims(x_dim=lon_dim, y_dim=lat_dim)
    da_filled = da_filled.rio.write_nodata(nodata, inplace=False)

    # Output filename
    if output_filename is None:
        safe_label = time_label.replace(":", "-").replace(" ", "_")
        output_filename = f"{variable}_{safe_label}_{uuid.uuid4().hex[:8]}.tif"
    if not output_filename.endswith(".tif"):
        output_filename += ".tif"

    data_dir = _ensure_data_dir()
    out_path = data_dir / output_filename

    # Write as Cloud-Optimized GeoTIFF
    da_filled.rio.to_raster(
        str(out_path),
        driver="GTiff",
        dtype="float32",
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        overviews="auto",
    )

    # Compute bounding box
    lats = da[lat_dim].values
    lons = da[lon_dim].values
    bbox = [
        float(np.nanmin(lons)),
        float(np.nanmin(lats)),
        float(np.nanmax(lons)),
        float(np.nanmax(lats)),
    ]

    cog_url = _rcmes_file_url(output_filename)
    logger.info("Exported COG: %s → %s", out_path, cog_url)

    return {
        "file_path": str(out_path),
        "cog_url": cog_url,
        "variable": variable,
        "bbox": bbox,
        "crs": "EPSG:4326",
        "shape": [int(da.shape[-2]), int(da.shape[-1])],
        "time_label": time_label,
        "output_filename": output_filename,
    }


# ─── Tool 2: Export as GeoJSON ────────────────────────────────────────────


@mcp.tool()
def export_climate_geojson(
    dataset_id: str,
    variable: str | None = None,
    statistic: str = "mean",
    output_filename: str | None = None,
    max_features: int = 50000,
) -> dict:
    """
    Export a climate dataset variable as a GeoJSON FeatureCollection.

    Each grid cell becomes a GeoJSON Point feature with the computed statistic
    as a property. Suitable for MMGIS vector or data layers.

    Args:
        dataset_id: ID of the dataset in the session (from load_climate_data).
        variable: Variable to export. Defaults to the first data variable.
        statistic: Statistic to compute over time — "mean", "max", "min",
                   "std", or "last". For datasets with no time dimension,
                   raw values are used.
        output_filename: Override output filename (must end in .geojson).
        max_features: Cap on number of features to avoid huge files. Grid cells
                      are sampled evenly if the dataset exceeds this limit.

    Returns:
        dict with keys:
            file_path     — absolute path on disk
            geojson_url   — URL to fetch the GeoJSON (use in push_layer_to_mmgis)
            variable      — variable exported
            statistic     — statistic computed
            feature_count — number of GeoJSON features written
            bbox          — [lon_min, lat_min, lon_max, lat_max]
    """
    import json

    import numpy as np

    try:
        ds = session_manager.get(dataset_id)
    except KeyError:
        raise ValueError(f"Dataset '{dataset_id}' not found in session.")

    if variable is None:
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError("Dataset has no data variables.")
        variable = data_vars[0]
    if variable not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not in dataset. Available: {list(ds.data_vars)}")

    da = ds[variable]

    # Resolve lat/lon dimension names
    lat_dim = next((d for d in da.dims if d in ("lat", "latitude")), None)
    lon_dim = next((d for d in da.dims if d in ("lon", "longitude")), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Cannot find lat/lon dimensions in {da.dims}")

    # Reduce over time if present
    if "time" in da.dims:
        agg_map = {
            "mean": da.mean(dim="time"),
            "max": da.max(dim="time"),
            "min": da.min(dim="time"),
            "std": da.std(dim="time"),
            "last": da.isel(time=-1),
        }
        if statistic not in agg_map:
            raise ValueError(
                f"Unknown statistic '{statistic}'. Use: mean, max, min, std, last."
            )
        da_2d = agg_map[statistic]
    else:
        da_2d = da
        statistic = "raw"

    values = da_2d.values  # shape (lat, lon)
    lats = da_2d[lat_dim].values
    lons = da_2d[lon_dim].values

    # Build flat list of (lat, lon, value), dropping NaN
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    flat_lats = lat_grid.ravel()
    flat_lons = lon_grid.ravel()
    flat_vals = values.ravel()

    mask = ~np.isnan(flat_vals)
    flat_lats = flat_lats[mask]
    flat_lons = flat_lons[mask]
    flat_vals = flat_vals[mask]

    total = len(flat_lats)

    # Subsample if too large
    if total > max_features:
        step = total // max_features
        flat_lats = flat_lats[::step]
        flat_lons = flat_lons[::step]
        flat_vals = flat_vals[::step]

    feature_count = len(flat_lats)

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": {variable: round(float(val), 4)},
        }
        for lat, lon, val in zip(flat_lats, flat_lons, flat_vals)
    ]

    geojson = {"type": "FeatureCollection", "features": features}

    # Output file
    if output_filename is None:
        output_filename = f"{variable}_{statistic}_{uuid.uuid4().hex[:8]}.geojson"
    if not output_filename.endswith(".geojson"):
        output_filename += ".geojson"

    data_dir = _ensure_data_dir()
    out_path = data_dir / output_filename

    with open(out_path, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))

    bbox = [
        float(np.nanmin(flat_lons)),
        float(np.nanmin(flat_lats)),
        float(np.nanmax(flat_lons)),
        float(np.nanmax(flat_lats)),
    ]
    geojson_url = _rcmes_file_url(output_filename)

    logger.info("Exported GeoJSON: %s (%d features)", out_path, feature_count)

    return {
        "file_path": str(out_path),
        "geojson_url": geojson_url,
        "variable": variable,
        "statistic": statistic,
        "feature_count": feature_count,
        "bbox": bbox,
        "output_filename": output_filename,
    }


# ─── Tool 3: Push layer to MMGIS ──────────────────────────────────────────


@mcp.tool()
def push_layer_to_mmgis(
    layer_name: str,
    data_url: str,
    layer_type: str = "tile",
    mmgis_url: str | None = None,
    mission: str | None = None,
    api_token: str | None = None,
    colormap: str = "rdbu_r",
    description: str = "",
    opacity: float = 0.8,
) -> dict:
    """
    Push a geospatial layer (COG tile or GeoJSON vector) into a running MMGIS instance.

    This adds or replaces a named layer in the MMGIS mission configuration.
    After success, open browser_url to see the layer on the interactive map.

    Args:
        layer_name: Display name of the layer in MMGIS (e.g. "Temperature Trend 2050").
        data_url: URL of the data to display:
                  - For "tile" layers: the cog_url from export_climate_geotiff.
                    TiTiler will stream this COG as raster tiles.
                  - For "vector" layers: the geojson_url from export_climate_geojson.
        layer_type: "tile" for raster COG layers (via TiTiler), or "vector" for
                    GeoJSON point/polygon layers. Default is "tile".
        mmgis_url: Base URL of the MMGIS instance (overrides MMGIS_URL env var).
        mission: MMGIS mission name (overrides MMGIS_MISSION env var).
        api_token: Long-term API token (overrides MMGIS_API_TOKEN env var).
        colormap: Matplotlib/TiTiler colormap name for tile layers (e.g.
                  "rdbu_r", "viridis", "plasma", "coolwarm"). Ignored for vector.
        description: Human-readable description shown in MMGIS layer panel.
        opacity: Layer opacity 0.0–1.0 (default 0.8).

    Returns:
        dict with keys:
            status       — "success" or "error"
            layer_name   — name of the pushed layer
            mission      — mission name
            version      — new MMGIS config version number
            browser_url  — URL to open in browser to view the layer in MMGIS
            message      — human-readable confirmation or error detail
    """
    from rcmes_mcp.utils import mmgis_client as mc

    resolved_url = mmgis_url or os.environ.get("MMGIS_URL", "http://localhost:2888")
    resolved_mission = mission or os.environ.get("MMGIS_MISSION", "climate")
    resolved_token = api_token or os.environ.get("MMGIS_API_TOKEN", "")
    titiler_external = os.environ.get("MMGIS_TITILER_EXTERNAL_URL", "http://localhost:8080")
    mmgis_external = os.environ.get("MMGIS_EXTERNAL_URL", resolved_url)

    # Ensure the mission exists
    if not mc.mission_exists(resolved_mission, url=resolved_url, token=resolved_token):
        logger.info("Mission '%s' not found — creating it.", resolved_mission)
        mc.add_mission(resolved_mission, url=resolved_url, token=resolved_token)

    # Fetch current config
    try:
        config = mc.get_mission_config(
            url=resolved_url, mission=resolved_mission, token=resolved_token
        )
    except Exception as exc:
        return {
            "status": "error",
            "layer_name": layer_name,
            "mission": resolved_mission,
            "message": f"Failed to fetch MMGIS config: {exc}",
        }

    # Build layer definition
    if layer_type == "tile":
        layer = mc.build_tile_layer(
            name=layer_name,
            cog_url=data_url,
            description=description,
            colormap=colormap,
            opacity=opacity,
            titiler_url=titiler_external,
        )
    elif layer_type == "vector":
        layer = mc.build_vector_layer(
            name=layer_name,
            geojson_url=data_url,
            description=description,
            opacity=opacity,
        )
    else:
        return {
            "status": "error",
            "layer_name": layer_name,
            "mission": resolved_mission,
            "message": f"Unknown layer_type '{layer_type}'. Use 'tile' or 'vector'.",
        }

    # Inject into config and push
    updated_config = mc.inject_layer(config, layer)
    try:
        result = mc.upsert_mission_config(
            updated_config,
            url=resolved_url,
            mission=resolved_mission,
            token=resolved_token,
        )
    except Exception as exc:
        return {
            "status": "error",
            "layer_name": layer_name,
            "mission": resolved_mission,
            "message": f"Failed to upsert MMGIS config: {exc}",
        }

    version = result.get("version", "?")
    browser_url = f"{mmgis_external.rstrip('/')}/?mission={resolved_mission}"

    logger.info(
        "Pushed layer '%s' to MMGIS mission '%s' v%s → %s",
        layer_name, resolved_mission, version, browser_url,
    )

    return {
        "status": "success",
        "layer_name": layer_name,
        "mission": resolved_mission,
        "version": version,
        "browser_url": browser_url,
        "message": (
            f"Layer '{layer_name}' is now live in MMGIS. "
            f"Open the map at: {browser_url}"
        ),
    }
