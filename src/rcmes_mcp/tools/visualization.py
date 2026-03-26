"""
Visualization Tools

MCP tools for generating climate data visualizations including
maps, time series plots, and Taylor diagrams.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger("rcmes.tools.visualization")

from rcmes_mcp.server import mcp
from rcmes_mcp.utils.session import session_manager

# Re-use thread-local for progress callbacks
from rcmes_mcp.tools.data_access import _thread_local


def _progress(step: int, total: int, detail: str):
    """Emit a progress event if a callback is registered."""
    cb = getattr(_thread_local, 'progress_callback', None)
    if cb:
        cb(step, total, detail)

# Simple PNG cache directory
_PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "rcmes_mcp_plot_cache"
_PLOT_CACHE_DIR.mkdir(exist_ok=True)


def _plot_cache_key(**kwargs) -> str:
    """Generate a cache key from plot parameters."""
    raw = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_cached_plot(cache_key: str) -> dict | None:
    """Return cached plot result if available."""
    png_path = _PLOT_CACHE_DIR / f"{cache_key}.png"
    meta_path = _PLOT_CACHE_DIR / f"{cache_key}.json"
    if png_path.exists() and meta_path.exists():
        with open(png_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["image_base64"] = img_base64
        meta["cached"] = True
        return meta
    return None


def _cache_plot(cache_key: str, fig, result: dict) -> None:
    """Save plot PNG and metadata to cache."""
    png_path = _PLOT_CACHE_DIR / f"{cache_key}.png"
    fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    meta_path = _PLOT_CACHE_DIR / f"{cache_key}.json"
    # Save metadata without image_base64
    meta = {k: v for k, v in result.items() if k != "image_base64"}
    with open(meta_path, "w") as f:
        json.dump(meta, f, default=str)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _save_fig(fig, filename: str) -> str:
    """Save figure to temporary file and return path."""
    temp_dir = Path(tempfile.gettempdir()) / "rcmes_mcp_plots"
    temp_dir.mkdir(exist_ok=True)
    filepath = temp_dir / filename
    fig.savefig(filepath, format="png", dpi=150, bbox_inches="tight")
    return str(filepath)


def _normalize_lons(data: xr.DataArray) -> xr.DataArray:
    """Convert longitudes from 0-360 to -180/180 if needed."""
    if float(data.lon.max()) > 180:
        data = data.assign_coords(
            lon=((data.lon + 180) % 360) - 180
        ).sortby("lon")
    return data


@mcp.tool()
def generate_map(
    dataset_id: str,
    time_index: int | str | None = None,
    title: str | None = None,
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict:
    """
    Generate a map visualization of spatial data.

    Args:
        dataset_id: ID of the dataset to visualize
        time_index: Time index (integer) or date string to plot. If None, plots temporal mean.
        title: Plot title (auto-generated if not provided)
        colormap: Matplotlib colormap name (viridis, RdBu_r, coolwarm, etc.)
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale

    Returns:
        Dictionary with base64-encoded image and file path
    """
    t0 = time.perf_counter()

    _progress(1, 4, "Checking plot cache...")
    # Check plot cache
    cache_key = _plot_cache_key(
        func="map", dataset_id=dataset_id, time_index=time_index,
        title=title, colormap=colormap, vmin=vmin, vmax=vmax,
    )
    cached = _get_cached_plot(cache_key)
    if cached:
        logger.debug(f"generate_map {dataset_id} served from cache", extra={"tool": "generate_map", "dataset_id": dataset_id})
        return cached

    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    try:
        _progress(2, 4, "Computing spatial data...")
        import matplotlib.pyplot as plt

        # Try to import cartopy for better maps
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            has_cartopy = True
        except ImportError:
            has_cartopy = False

        # Get data to plot
        if isinstance(ds, xr.DataArray):
            data = ds
        else:
            var_name = metadata.variable if (metadata.variable and metadata.variable in ds.data_vars) else list(ds.data_vars)[0]
            data = ds[var_name]

        # Handle time dimension
        if "time" in data.dims:
            if time_index is None:
                # Plot temporal mean
                plot_data = data.mean(dim="time").compute()
                time_label = "Temporal Mean"
            elif isinstance(time_index, int):
                plot_data = data.isel(time=time_index).compute()
                time_label = str(data.time.values[time_index])[:10]
            else:
                plot_data = data.sel(time=time_index, method="nearest").compute()
                time_label = time_index
        else:
            plot_data = data.compute()
            time_label = ""

        # Normalize longitudes from 0-360 to -180/180
        plot_data = _normalize_lons(plot_data)

        # Detect if this is a global-extent map
        lon_range = float(plot_data.lon.max()) - float(plot_data.lon.min())
        is_global = lon_range > 300

        # Create figure
        _progress(3, 4, "Rendering map...")
        if has_cartopy:
            fig, ax = plt.subplots(
                figsize=(14, 7) if is_global else (12, 8),
                subplot_kw={"projection": ccrs.PlateCarree()}
            )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")
            if not is_global:
                ax.add_feature(cfeature.STATES, linewidth=0.2)

            im = ax.pcolormesh(
                plot_data.lon,
                plot_data.lat,
                plot_data.values,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
            )
            if is_global:
                ax.set_global()
            else:
                ax.set_extent([
                    float(plot_data.lon.min()),
                    float(plot_data.lon.max()),
                    float(plot_data.lat.min()),
                    float(plot_data.lat.max()),
                ])
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.pcolormesh(
                plot_data.lon,
                plot_data.lat,
                plot_data.values,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(data.attrs.get("units", metadata.variable or ""))

        # Set title
        if title:
            ax.set_title(title)
        else:
            model_str = f" ({metadata.model})" if metadata.model else ""
            ax.set_title(f"{metadata.variable}{model_str}\n{time_label}")

        # Save, encode, and cache
        _progress(4, 4, "Encoding image...")
        filename = f"map_{dataset_id}.png"
        filepath = _save_fig(fig, filename)
        img_base64 = _fig_to_base64(fig)

        result = {
            "success": True,
            "image_base64": img_base64,
            "file_path": filepath,
            "dataset_id": dataset_id,
            "time": time_label,
        }
        _cache_plot(cache_key, fig, result)
        plt.close(fig)

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"generate_map {dataset_id} in {duration_ms}ms", extra={"tool": "generate_map", "dataset_id": dataset_id, "duration_ms": duration_ms})

        return result

    except Exception as e:
        logger.exception(f"generate_map failed for {dataset_id}", extra={"tool": "generate_map", "error": str(e)})
        return {"error": f"Failed to generate map: {str(e)}"}


@mcp.tool()
def generate_timeseries_plot(
    dataset_ids: list[str],
    labels: list[str] | None = None,
    title: str | None = None,
    show_trend: bool = False,
    show_uncertainty: bool = False,
) -> dict:
    """
    Generate a time series plot for one or more datasets.

    Args:
        dataset_ids: List of dataset IDs to plot
        labels: Labels for each dataset (auto-generated if not provided)
        title: Plot title
        show_trend: Show linear trend lines
        show_uncertainty: Show uncertainty envelope (std dev)

    Returns:
        Dictionary with base64-encoded image and file path
    """
    if not dataset_ids:
        return {"error": "No dataset IDs provided"}

    t0 = time.perf_counter()
    total_steps = len(dataset_ids) + 2  # data steps + render + encode

    _progress(1, total_steps, "Checking plot cache...")
    # Check plot cache
    cache_key = _plot_cache_key(
        func="timeseries", dataset_ids=dataset_ids, labels=labels,
        title=title, show_trend=show_trend, show_uncertainty=show_uncertainty,
    )
    cached = _get_cached_plot(cache_key)
    if cached:
        return cached

    try:
        import matplotlib.pyplot as plt
        from scipy import stats

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, dataset_id in enumerate(dataset_ids):
            _progress(i + 2, total_steps, f"Processing dataset {i+1}/{len(dataset_ids)}...")
            try:
                ds = session_manager.get(dataset_id)
                metadata = session_manager.get_metadata(dataset_id)
            except KeyError:
                continue

            # Get data
            if isinstance(ds, xr.DataArray):
                data = ds
            else:
                var_name = metadata.variable if (metadata.variable and metadata.variable in ds.data_vars) else list(ds.data_vars)[0]
                data = ds[var_name]

            # Calculate spatial mean (and std if needed) in a single compute pass
            needs_spatial_reduce = "lat" in data.dims and "lon" in data.dims
            if needs_spatial_reduce:
                weights = np.cos(np.deg2rad(data.lat))
                mean_data = data.weighted(weights).mean(dim=["lat", "lon"])
                if show_uncertainty:
                    std_data = data.std(dim=["lat", "lon"])
                    import dask
                    ts, std_ts = dask.compute(mean_data, std_data)
                else:
                    ts = mean_data.compute()
                    std_ts = None
            else:
                ts = data.compute()
                std_ts = None

            # Generate label
            if labels and i < len(labels):
                label = labels[i]
            else:
                model = metadata.model or ""
                scenario = metadata.scenario or ""
                label = f"{model} {scenario}".strip() or dataset_id

            # Plot
            ax.plot(ts.time.values, ts.values, label=label, linewidth=1.5)

            # Add trend line
            if show_trend:
                x = np.arange(len(ts))
                y = ts.values
                mask = ~np.isnan(y)
                if mask.sum() > 2:
                    slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])
                    trend_line = slope * x + intercept
                    ax.plot(
                        ts.time.values,
                        trend_line,
                        linestyle="--",
                        alpha=0.7,
                        label=f"{label} trend",
                    )

            # Add uncertainty envelope
            if std_ts is not None:
                ax.fill_between(
                    ts.time.values,
                    ts.values - std_ts.values,
                    ts.values + std_ts.values,
                    alpha=0.2,
                )

        ax.set_xlabel("Time")
        ax.set_ylabel(data.attrs.get("units", ""))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title)

        # Save, encode, and cache
        _progress(total_steps, total_steps, "Encoding image...")
        filename = f"timeseries_{dataset_ids[0]}.png"
        filepath = _save_fig(fig, filename)
        img_base64 = _fig_to_base64(fig)

        result = {
            "success": True,
            "image_base64": img_base64,
            "file_path": filepath,
            "dataset_ids": dataset_ids,
        }
        _cache_plot(cache_key, fig, result)
        plt.close(fig)

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"generate_timeseries_plot {dataset_ids} in {duration_ms}ms", extra={"tool": "generate_timeseries_plot", "duration_ms": duration_ms})

        return result

    except Exception as e:
        logger.exception(f"generate_timeseries_plot failed", extra={"tool": "generate_timeseries_plot", "error": str(e)})
        return {"error": f"Failed to generate time series plot: {str(e)}"}


@mcp.tool()
def generate_comparison_map(
    dataset_ids: list[str],
    labels: list[str] | None = None,
    title: str | None = None,
    colormap: str = "viridis",
) -> dict:
    """
    Generate side-by-side comparison maps for multiple datasets.

    Args:
        dataset_ids: List of dataset IDs to compare (2-4 datasets)
        labels: Labels for each subplot
        title: Overall figure title
        colormap: Matplotlib colormap

    Returns:
        Dictionary with base64-encoded image and file path
    """
    if not dataset_ids or len(dataset_ids) < 2:
        return {"error": "Need at least 2 dataset IDs for comparison"}

    if len(dataset_ids) > 4:
        return {"error": "Maximum 4 datasets for comparison"}

    try:
        import matplotlib.pyplot as plt

        # Determine subplot layout
        n = len(dataset_ids)
        if n == 2:
            nrows, ncols = 1, 2
        elif n == 3:
            nrows, ncols = 1, 3
        else:
            nrows, ncols = 2, 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.atleast_1d(axes).flatten()

        # Find common color scale
        all_data = []
        for dataset_id in dataset_ids:
            ds = session_manager.get(dataset_id)
            metadata = session_manager.get_metadata(dataset_id)
            if isinstance(ds, xr.DataArray):
                data = ds
            else:
                var_name = metadata.variable if (metadata.variable and metadata.variable in ds.data_vars) else list(ds.data_vars)[0]
                data = ds[var_name]

            if "time" in data.dims:
                data = data.mean(dim="time")

            all_data.append(_normalize_lons(data.compute()))

        vmin = min(float(d.min()) for d in all_data)
        vmax = max(float(d.max()) for d in all_data)

        # Plot each dataset
        for i, (dataset_id, data) in enumerate(zip(dataset_ids, all_data)):
            ax = axes[i]
            metadata = session_manager.get_metadata(dataset_id)

            im = ax.pcolormesh(
                data.lon,
                data.lat,
                data.values,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
            )

            if labels and i < len(labels):
                ax.set_title(labels[i])
            else:
                ax.set_title(f"{metadata.model or ''} {metadata.scenario or ''}".strip())

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        # Hide unused subplots
        for i in range(len(dataset_ids), len(axes)):
            axes[i].set_visible(False)

        # Add colorbar
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label=metadata.variable or "")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        # Save and encode
        filename = f"comparison_{dataset_ids[0]}.png"
        filepath = _save_fig(fig, filename)
        img_base64 = _fig_to_base64(fig)
        plt.close(fig)

        return {
            "success": True,
            "image_base64": img_base64,
            "file_path": filepath,
            "dataset_ids": dataset_ids,
        }

    except Exception as e:
        return {"error": f"Failed to generate comparison map: {str(e)}"}


@mcp.tool()
def generate_taylor_diagram(
    model_dataset_ids: list[str],
    reference_dataset_id: str,
    labels: list[str] | None = None,
    title: str = "Taylor Diagram",
) -> dict:
    """
    Generate a Taylor diagram for model evaluation.

    Taylor diagrams display the correlation, standard deviation ratio,
    and centered RMS difference between model and reference data.

    Args:
        model_dataset_ids: List of model dataset IDs to evaluate
        reference_dataset_id: Reference/observation dataset ID
        labels: Labels for each model
        title: Diagram title

    Returns:
        Dictionary with base64-encoded image and file path
    """
    try:
        import matplotlib.pyplot as plt

        # Get reference data
        ref_ds = session_manager.get(reference_dataset_id)
        if isinstance(ref_ds, xr.DataArray):
            ref_data = ref_ds
        else:
            ref_data = ref_ds[list(ref_ds.data_vars)[0]]

        if "time" in ref_data.dims:
            ref_data = ref_data.mean(dim="time").compute()
        else:
            ref_data = ref_data.compute()

        ref_std = float(ref_data.std())
        ref_values = ref_data.values.flatten()
        ref_values = ref_values[~np.isnan(ref_values)]

        # Calculate statistics for each model
        model_stats = []
        for i, dataset_id in enumerate(model_dataset_ids):
            try:
                ds = session_manager.get(dataset_id)
                metadata = session_manager.get_metadata(dataset_id)
            except KeyError:
                continue

            if isinstance(ds, xr.DataArray):
                model_data = ds
            else:
                model_data = ds[list(ds.data_vars)[0]]

            if "time" in model_data.dims:
                model_data = model_data.mean(dim="time").compute()
            else:
                model_data = model_data.compute()

            model_values = model_data.values.flatten()
            model_values = model_values[~np.isnan(model_values)]

            # Match sizes
            min_len = min(len(ref_values), len(model_values))
            r = ref_values[:min_len]
            m = model_values[:min_len]

            # Calculate correlation
            correlation = np.corrcoef(r, m)[0, 1]

            # Calculate normalized standard deviation
            model_std = float(model_data.std())
            std_ratio = model_std / ref_std

            # Get label
            if labels and i < len(labels):
                label = labels[i]
            else:
                label = f"{metadata.model or ''} {metadata.scenario or ''}".strip() or dataset_id

            model_stats.append({
                "label": label,
                "correlation": correlation,
                "std_ratio": std_ratio,
            })

        if not model_stats:
            return {"error": "No valid model datasets to plot"}

        # Create Taylor diagram
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

        # Reference point
        ax.plot(0, 1, "ko", markersize=10, label="Reference")

        # Plot models
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_stats)))
        for stat, color in zip(model_stats, colors):
            theta = np.arccos(stat["correlation"])
            r = stat["std_ratio"]
            ax.plot(theta, r, "o", markersize=10, color=color, label=stat["label"])

        # Configure polar plot for Taylor diagram
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2)

        # Add correlation labels
        correlation_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        ax.set_thetagrids(
            [np.degrees(np.arccos(c)) for c in correlation_ticks],
            labels=[str(c) for c in correlation_ticks],
        )

        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

        plt.tight_layout()

        # Save and encode
        filename = f"taylor_{reference_dataset_id}.png"
        filepath = _save_fig(fig, filename)
        img_base64 = _fig_to_base64(fig)
        plt.close(fig)

        return {
            "success": True,
            "image_base64": img_base64,
            "file_path": filepath,
            "model_statistics": model_stats,
            "reference_dataset_id": reference_dataset_id,
        }

    except Exception as e:
        return {"error": f"Failed to generate Taylor diagram: {str(e)}"}


@mcp.tool()
def generate_country_map(
    dataset_id: str,
    country_name: str | None = None,
    time_index: int | str | None = None,
    title: str | None = None,
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict:
    """
    Generate a map visualization with country boundaries highlighted.

    Useful for plotting ETCCDI index maps masked to a specific country.

    Args:
        dataset_id: ID of the dataset to visualize
        country_name: Country to highlight with bold boundary (optional)
        time_index: Time index (integer) or date string to plot. If None, plots temporal mean.
        title: Plot title (auto-generated if not provided)
        colormap: Matplotlib colormap name
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale

    Returns:
        Dictionary with base64-encoded image and file path
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    try:
        import matplotlib.pyplot as plt

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError:
            return {"error": "cartopy is required for generate_country_map. Install with: pip install cartopy"}

        # Get data to plot
        if isinstance(ds, xr.DataArray):
            data = ds
        else:
            var_name = metadata.variable if (metadata.variable and metadata.variable in ds.data_vars) else list(ds.data_vars)[0]
            data = ds[var_name]

        # Handle time dimension
        if "time" in data.dims:
            if time_index is None:
                plot_data = data.mean(dim="time").compute()
                time_label = "Temporal Mean"
            elif isinstance(time_index, int):
                plot_data = data.isel(time=time_index).compute()
                time_label = str(data.time.values[time_index])[:10]
            else:
                plot_data = data.sel(time=time_index, method="nearest").compute()
                time_label = time_index
        else:
            plot_data = data.compute()
            time_label = ""

        # Normalize longitudes from 0-360 to -180/180
        plot_data = _normalize_lons(plot_data)

        # Detect global extent
        lon_range = float(plot_data.lon.max()) - float(plot_data.lon.min())
        is_global = lon_range > 300

        # Create figure with cartopy
        fig, ax = plt.subplots(
            figsize=(14, 7) if is_global else (12, 8),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        # Add base features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--")

        # Plot data
        im = ax.pcolormesh(
            plot_data.lon,
            plot_data.lat,
            plot_data.values,
            transform=ccrs.PlateCarree(),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
        )

        # Highlight country boundary if specified
        if country_name:
            try:
                from rcmes_mcp.tools.processing import _get_country_boundaries

                gdf = _get_country_boundaries()
                country_row = None
                for col in ["NAME", "name", "ADMIN", "admin"]:
                    if col in gdf.columns:
                        matches = gdf[gdf[col].str.lower() == country_name.lower()]
                        if len(matches) > 0:
                            country_row = matches.iloc[0]
                            break

                if country_row is not None:
                    from cartopy.feature import ShapelyFeature

                    feature = ShapelyFeature(
                        [country_row.geometry],
                        ccrs.PlateCarree(),
                        facecolor="none",
                        edgecolor="black",
                        linewidth=2.5,
                    )
                    ax.add_feature(feature)
            except Exception:
                pass  # Non-critical: skip country highlight if it fails

        # Set extent based on data
        if is_global:
            ax.set_global()
        else:
            ax.set_extent([
                float(plot_data.lon.min()),
                float(plot_data.lon.max()),
                float(plot_data.lat.min()),
                float(plot_data.lat.max()),
            ])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(data.attrs.get("units", metadata.variable or ""))

        # Set title
        if title:
            ax.set_title(title)
        else:
            model_str = f" ({metadata.model})" if metadata.model else ""
            country_str = f" - {country_name}" if country_name else ""
            ax.set_title(f"{metadata.variable}{model_str}{country_str}\n{time_label}")

        # Save and encode
        filename = f"country_map_{dataset_id}.png"
        filepath = _save_fig(fig, filename)
        img_base64 = _fig_to_base64(fig)
        plt.close(fig)

        return {
            "success": True,
            "image_base64": img_base64,
            "file_path": filepath,
            "dataset_id": dataset_id,
            "country": country_name,
            "time": time_label,
        }

    except Exception as e:
        return {"error": f"Failed to generate country map: {str(e)}"}


@mcp.tool()
def generate_histogram(
    dataset_id: str,
    bins: int = 50,
    title: str | None = None,
) -> dict:
    """
    Generate a histogram of data values.

    Args:
        dataset_id: ID of the dataset
        bins: Number of histogram bins
        title: Plot title

    Returns:
        Dictionary with base64-encoded image and statistics
    """
    try:
        import matplotlib.pyplot as plt

        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)

        if isinstance(ds, xr.DataArray):
            data = ds
        else:
            var_name = metadata.variable if (metadata.variable and metadata.variable in ds.data_vars) else list(ds.data_vars)[0]
            data = ds[var_name]

        values = data.values.flatten()
        values = values[~np.isnan(values)]

        fig, ax = plt.subplots(figsize=(10, 6))
        counts, bin_edges, _ = ax.hist(values, bins=bins, edgecolor="black", alpha=0.7)

        ax.set_xlabel(data.attrs.get("units", metadata.variable or "Value"))
        ax.set_ylabel("Frequency")

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Distribution of {metadata.variable}")

        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
        ax.legend()

        # Save and encode
        filename = f"histogram_{dataset_id}.png"
        filepath = _save_fig(fig, filename)
        img_base64 = _fig_to_base64(fig)
        plt.close(fig)

        return {
            "success": True,
            "image_base64": img_base64,
            "file_path": filepath,
            "statistics": {
                "mean": float(mean_val),
                "std": float(std_val),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_samples": len(values),
            },
        }

    except Exception as e:
        return {"error": f"Failed to generate histogram: {str(e)}"}
