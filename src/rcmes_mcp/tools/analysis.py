"""
Analysis Tools

MCP tools for climate data analysis including statistics, trends,
and model evaluation metrics.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import xarray as xr
from scipy import stats

from rcmes_mcp.server import mcp
from rcmes_mcp.utils.session import session_manager
from rcmes_mcp.utils.validation import validate_date_range

logger = logging.getLogger("rcmes.tools.analysis")

# Re-use the same thread-local as data_access for progress callbacks
from rcmes_mcp.tools.data_access import _thread_local


def _progress(step: int, total: int, detail: str):
    """Emit a progress event if a callback is registered."""
    cb = getattr(_thread_local, 'progress_callback', None)
    if cb:
        cb(step, total, detail)


def _ensure_materialized(dataset_id: str):
    """Wait for background materialization if in progress, with progress updates."""
    from rcmes_mcp.tools.data_access import _materialization_events
    event = _materialization_events.get(dataset_id)
    if event is None:
        return  # Already materialized or no pending work
    _progress(0, 0, "Downloading data from S3 (one-time)...")
    # Poll with progress heartbeats so the UI stays updated
    while not event.wait(timeout=3.0):
        _progress(0, 0, "Still downloading from S3...")


@mcp.tool()
def calculate_statistics(
    dataset_id: str,
    statistic: str = "all",
) -> dict:
    """
    Calculate summary statistics for a dataset.

    Args:
        dataset_id: ID of the dataset
        statistic: Which statistic to compute - "mean", "std", "min", "max",
                   "percentiles", or "all" (default)

    Returns:
        Dictionary with computed statistics
    """
    _ensure_materialized(dataset_id)

    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get the main data variable
    if isinstance(ds, xr.DataArray):
        data = ds
    else:
        var_name = metadata.variable if (metadata.variable and metadata.variable in ds.data_vars) else list(ds.data_vars)[0]
        data = ds[var_name]

    results = {
        "dataset_id": dataset_id,
        "variable": metadata.variable,
    }

    try:
        if statistic == "all":
            # Single compute pass for all basic stats
            _progress(1, 3, "Computing mean, std, min, max...")
            import dask
            mean_val, std_val, min_val, max_val = dask.compute(
                data.mean(), data.std(), data.min(), data.max()
            )
            results["mean"] = float(mean_val)
            results["std"] = float(std_val)
            results["min"] = float(min_val)
            results["max"] = float(max_val)

            # Compute percentiles (subsample for large datasets)
            _progress(2, 3, "Computing percentiles...")
            sample = data.values.flatten()
            if len(sample) > 1000000:
                sample = np.random.choice(sample[~np.isnan(sample)], 1000000)
            results["percentiles"] = {
                "p5": float(np.nanpercentile(sample, 5)),
                "p25": float(np.nanpercentile(sample, 25)),
                "p50": float(np.nanpercentile(sample, 50)),
                "p75": float(np.nanpercentile(sample, 75)),
                "p95": float(np.nanpercentile(sample, 95)),
            }
            _progress(3, 3, "Statistics complete")
        else:
            if statistic == "mean":
                results["mean"] = float(data.mean().compute())
            elif statistic == "std":
                results["std"] = float(data.std().compute())
            elif statistic == "min":
                results["min"] = float(data.min().compute())
            elif statistic == "max":
                results["max"] = float(data.max().compute())
            elif statistic == "percentiles":
                sample = data.values.flatten()
                if len(sample) > 1000000:
                    sample = np.random.choice(sample[~np.isnan(sample)], 1000000)
                results["percentiles"] = {
                    "p5": float(np.nanpercentile(sample, 5)),
                    "p25": float(np.nanpercentile(sample, 25)),
                    "p50": float(np.nanpercentile(sample, 50)),
                    "p75": float(np.nanpercentile(sample, 75)),
                    "p95": float(np.nanpercentile(sample, 95)),
                }

    except Exception as e:
        logger.exception(f"calculate_statistics failed for {dataset_id}", extra={"tool": "calculate_statistics", "error": str(e)})
        return {"error": f"Failed to compute statistics: {str(e)}"}

    logger.info(f"calculate_statistics {dataset_id} → {statistic}", extra={"tool": "calculate_statistics", "dataset_id": dataset_id})
    return results


@mcp.tool()
def calculate_climatology(
    dataset_id: str,
    period: str = "monthly",
) -> dict:
    """
    Calculate climatological mean for each period (day, month, or season).

    Args:
        dataset_id: ID of the dataset
        period: Climatology period - "daily", "monthly", "seasonal"

    Returns:
        New dataset_id containing climatology
    """
    _ensure_materialized(dataset_id)
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    try:
        _progress(1, 2, f"Computing {period} climatology...")
        if period == "daily":
            climatology = ds.groupby("time.dayofyear").mean()
        elif period == "monthly":
            climatology = ds.groupby("time.month").mean()
        elif period == "seasonal":
            climatology = ds.groupby("time.season").mean()
        else:
            return {"error": f"Invalid period '{period}'. Use: daily, monthly, seasonal"}
        _progress(2, 2, "Climatology complete")

    except Exception as e:
        return {"error": f"Failed to calculate climatology: {str(e)}"}

    new_id = session_manager.store(
        data=climatology,
        source=metadata.source,
        variable=f"{metadata.variable}_climatology",
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"{period.capitalize()} climatology of {dataset_id}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "period": period,
    }


@mcp.tool()
def calculate_trend(
    dataset_id: str,
    method: str = "linear",
) -> dict:
    """
    Calculate temporal trend with statistical significance.

    Args:
        dataset_id: ID of the dataset
        method: Trend method - "linear" (OLS regression)

    Returns:
        Trend statistics including slope, p-value, and confidence intervals
    """
    _ensure_materialized(dataset_id)
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get the main data variable
    if isinstance(ds, xr.DataArray):
        data = ds
    else:
        var_name = metadata.variable if (metadata.variable and metadata.variable in ds.data_vars) else list(ds.data_vars)[0]
        data = ds[var_name]

    try:
        _progress(1, 3, "Computing area-weighted spatial mean...")
        # Calculate area-weighted spatial mean time series
        weights = np.cos(np.deg2rad(data.lat))
        weights = weights / weights.sum()

        if "lat" in data.dims and "lon" in data.dims:
            time_series = data.weighted(weights).mean(dim=["lat", "lon"])
        else:
            time_series = data.mean(dim=[d for d in data.dims if d != "time"])

        # Convert to numpy
        _progress(2, 3, "Running linear regression...")
        y = time_series.values
        x = np.arange(len(y))

        # Remove NaN values
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(y) < 3:
            return {"error": "Insufficient data points for trend calculation"}

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Detect temporal frequency from time coordinate
        times = time_series.time.values
        if len(times) >= 2:
            # Calculate median time step in days
            time_diffs = np.diff(times.astype("datetime64[D]").astype(float))
            median_step_days = float(np.median(time_diffs))

            if median_step_days > 300:
                # Annual data: each step ~1 year, 10 steps per decade
                steps_per_decade = 10
                freq_label = "annual"
            elif median_step_days > 25:
                # Monthly data: each step ~1 month, 120 steps per decade
                steps_per_decade = 120
                freq_label = "monthly"
            else:
                # Daily data: each step ~1 day, 3652.5 steps per decade
                steps_per_decade = 3652.5
                freq_label = "daily"
        else:
            steps_per_decade = 3652.5
            freq_label = "daily"

        # Convert slope to per-decade
        slope_per_decade = slope * steps_per_decade

        # Confidence interval (95%)
        ci_95 = 1.96 * std_err * steps_per_decade
        _progress(3, 3, "Trend analysis complete")

    except Exception as e:
        logger.exception(f"calculate_trend failed for {dataset_id}", extra={"tool": "calculate_trend", "error": str(e)})
        return {"error": f"Failed to calculate trend: {str(e)}"}

    logger.info(
        f"calculate_trend {dataset_id}: {slope_per_decade:.3f}/decade, p={p_value:.4f}",
        extra={"tool": "calculate_trend", "dataset_id": dataset_id},
    )

    return {
        "dataset_id": dataset_id,
        "variable": metadata.variable,
        "trend": {
            "slope_per_decade": float(slope_per_decade),
            "unit": f"{data.attrs.get('units', 'units')} per decade",
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "confidence_interval_95": {
                "lower": float(slope_per_decade - ci_95),
                "upper": float(slope_per_decade + ci_95),
            },
            "significant": p_value < 0.05,
        },
        "interpretation": (
            f"The trend is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} "
            f"(p={p_value:.4f}). "
            f"The variable changes by {slope_per_decade:.3f} {data.attrs.get('units', 'units')} per decade."
        ),
    }


@mcp.tool()
def calculate_regional_mean(
    dataset_id: str,
    area_weighted: bool = True,
) -> dict:
    """
    Calculate regional mean time series with optional area weighting.

    Args:
        dataset_id: ID of the dataset
        area_weighted: Apply latitude-based area weighting (default: True)

    Returns:
        New dataset_id containing the regional mean time series
    """
    _ensure_materialized(dataset_id)
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get the main data variable
    if isinstance(ds, xr.DataArray):
        data = ds
    else:
        var_name = metadata.variable if (metadata.variable and metadata.variable in ds.data_vars) else list(ds.data_vars)[0]
        data = ds[var_name]

    try:
        _progress(1, 2, "Computing area-weighted regional mean...")
        if area_weighted and "lat" in data.dims:
            # Calculate weights based on latitude
            weights = np.cos(np.deg2rad(data.lat))
            regional_mean = data.weighted(weights).mean(dim=["lat", "lon"])
        else:
            regional_mean = data.mean(dim=["lat", "lon"])

        # Eagerly compute the regional mean (1D time series) so downstream
        # tools don't need to re-pull from S3.
        regional_mean = regional_mean.compute()
        _progress(2, 2, "Regional mean complete")

    except Exception as e:
        return {"error": f"Failed to calculate regional mean: {str(e)}"}

    new_id = session_manager.store(
        data=regional_mean,
        source=metadata.source,
        variable=f"{metadata.variable}_regional_mean",
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"{'Area-weighted ' if area_weighted else ''}regional mean of {dataset_id}",
    )

    # Summary stats are instant since regional_mean is already computed
    mean_val = float(regional_mean.mean())
    std_val = float(regional_mean.std())

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "area_weighted": area_weighted,
        "summary": {
            "mean": mean_val,
            "std": std_val,
            "time_steps": regional_mean.sizes.get("time", len(regional_mean)),
        },
    }


@mcp.tool()
def calculate_bias(
    model_dataset_id: str,
    reference_dataset_id: str,
    metric: str = "mean",
) -> dict:
    """
    Calculate bias between model and reference dataset.

    Args:
        model_dataset_id: ID of the model dataset
        reference_dataset_id: ID of the reference/observation dataset
        metric: Bias type - "mean" (average difference), "absolute" (MAE), "relative" (%)

    Returns:
        Bias statistics and new dataset_id with spatial bias field
    """
    try:
        model_ds = session_manager.get(model_dataset_id)
        ref_ds = session_manager.get(reference_dataset_id)
        model_meta = session_manager.get_metadata(model_dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get data arrays
    if isinstance(model_ds, xr.DataArray):
        model_data = model_ds
    else:
        var_name = model_meta.variable or list(model_ds.data_vars)[0]
        model_data = model_ds[var_name]

    if isinstance(ref_ds, xr.DataArray):
        ref_data = ref_ds
    else:
        ref_data = ref_ds[list(ref_ds.data_vars)[0]]

    try:
        # Calculate temporal mean
        model_mean = model_data.mean(dim="time")
        ref_mean = ref_data.mean(dim="time")

        if metric == "mean":
            bias = model_mean - ref_mean
            bias_label = "Mean Bias"
        elif metric == "absolute":
            bias = np.abs(model_mean - ref_mean)
            bias_label = "Mean Absolute Error"
        elif metric == "relative":
            bias = (model_mean - ref_mean) / ref_mean * 100
            bias_label = "Relative Bias (%)"
        else:
            return {"error": f"Invalid metric '{metric}'. Use: mean, absolute, relative"}

        # Calculate overall statistics in a single compute pass
        import dask
        spatial_mean_bias, spatial_rmse = dask.compute(
            bias.mean(),
            np.sqrt(((model_mean - ref_mean) ** 2).mean()),
        )
        spatial_mean_bias = float(spatial_mean_bias)
        spatial_rmse = float(spatial_rmse)

    except Exception as e:
        return {"error": f"Failed to calculate bias: {str(e)}"}

    new_id = session_manager.store(
        data=bias,
        source="computed",
        variable=f"bias_{metric}",
        model=model_meta.model,
        scenario=model_meta.scenario,
        description=f"{bias_label} between {model_dataset_id} and {reference_dataset_id}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "model_dataset_id": model_dataset_id,
        "reference_dataset_id": reference_dataset_id,
        "metric": metric,
        "statistics": {
            "spatial_mean_bias": spatial_mean_bias,
            "spatial_rmse": spatial_rmse,
        },
    }


@mcp.tool()
def calculate_correlation(
    dataset1_id: str,
    dataset2_id: str,
    correlation_type: str = "temporal",
) -> dict:
    """
    Calculate correlation between two datasets.

    Args:
        dataset1_id: ID of first dataset
        dataset2_id: ID of second dataset
        correlation_type: "temporal" (correlation at each grid point over time)
                         or "spatial" (correlation at each time step over space)

    Returns:
        Correlation statistics and new dataset_id with correlation field
    """
    try:
        ds1 = session_manager.get(dataset1_id)
        ds2 = session_manager.get(dataset2_id)
        meta1 = session_manager.get_metadata(dataset1_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get data arrays
    if isinstance(ds1, xr.DataArray):
        data1 = ds1
    else:
        data1 = ds1[list(ds1.data_vars)[0]]

    if isinstance(ds2, xr.DataArray):
        data2 = ds2
    else:
        data2 = ds2[list(ds2.data_vars)[0]]

    try:
        if correlation_type == "temporal":
            # Correlation over time at each grid point
            correlation = xr.corr(data1, data2, dim="time")
        elif correlation_type == "spatial":
            # Correlation over space at each time step
            correlation = xr.corr(data1, data2, dim=["lat", "lon"])
        else:
            return {"error": f"Invalid correlation_type '{correlation_type}'"}

        # Overall statistics
        if correlation_type == "temporal":
            mean_corr = float(correlation.mean().compute())
        else:
            mean_corr = float(correlation.mean().compute())

    except Exception as e:
        return {"error": f"Failed to calculate correlation: {str(e)}"}

    new_id = session_manager.store(
        data=correlation,
        source="computed",
        variable=f"{correlation_type}_correlation",
        model=meta1.model,
        scenario=meta1.scenario,
        description=f"{correlation_type.capitalize()} correlation between {dataset1_id} and {dataset2_id}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "dataset1_id": dataset1_id,
        "dataset2_id": dataset2_id,
        "correlation_type": correlation_type,
        "mean_correlation": mean_corr,
    }


@mcp.tool()
def calculate_rmse(
    model_dataset_id: str,
    reference_dataset_id: str,
) -> dict:
    """
    Calculate Root Mean Square Error between model and reference.

    Args:
        model_dataset_id: ID of the model dataset
        reference_dataset_id: ID of the reference dataset

    Returns:
        RMSE statistics
    """
    try:
        model_ds = session_manager.get(model_dataset_id)
        ref_ds = session_manager.get(reference_dataset_id)
        model_meta = session_manager.get_metadata(model_dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get data arrays
    if isinstance(model_ds, xr.DataArray):
        model_data = model_ds
    else:
        model_data = model_ds[list(model_ds.data_vars)[0]]

    if isinstance(ref_ds, xr.DataArray):
        ref_data = ref_ds
    else:
        ref_data = ref_ds[list(ref_ds.data_vars)[0]]

    try:
        # Calculate squared differences
        squared_diff = (model_data - ref_data) ** 2

        # RMSE at each grid point (temporal RMSE)
        temporal_rmse = np.sqrt(squared_diff.mean(dim="time"))

        # Overall RMSE
        overall_rmse = float(np.sqrt(squared_diff.mean()).compute())

    except Exception as e:
        return {"error": f"Failed to calculate RMSE: {str(e)}"}

    new_id = session_manager.store(
        data=temporal_rmse,
        source="computed",
        variable="rmse",
        model=model_meta.model,
        scenario=model_meta.scenario,
        description=f"RMSE between {model_dataset_id} and {reference_dataset_id}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "model_dataset_id": model_dataset_id,
        "reference_dataset_id": reference_dataset_id,
        "overall_rmse": overall_rmse,
        "units": model_data.attrs.get("units", "unknown"),
    }


def _get_main_array(ds, metadata) -> xr.DataArray:
    """Pull the main DataArray from a Dataset, or return as-is if already a DataArray."""
    if isinstance(ds, xr.DataArray):
        return ds
    if metadata.variable and metadata.variable in ds.data_vars:
        return ds[metadata.variable]
    # Fall back to first/closest match (e.g., '<var>_zscore', '<var>_anomaly')
    for v in ds.data_vars:
        if metadata.variable and v.startswith(metadata.variable):
            return ds[v]
    return ds[list(ds.data_vars)[0]]


@mcp.tool()
def detect_extreme_events(
    dataset_id: str,
    sigma_threshold: float = 2.0,
    min_duration_days: int = 1,
    min_area_cells: int = 1,
    direction: str = "both",
    max_events: int = 50,
) -> dict:
    """
    Detect spatiotemporal extreme events using 3D connected-component labeling.

    Expects a STANDARDIZED ANOMALY field (run calculate_standardized_anomaly first).
    Finds contiguous space-time blobs where |z| exceeds sigma_threshold and returns
    an event catalogue (not a field).

    Args:
        dataset_id: ID of a standardized-anomaly dataset (z-scores)
        sigma_threshold: |z| threshold for extreme cells (default 2.0)
        min_duration_days: Drop events shorter than this many days
        min_area_cells: Drop events smaller than this many grid cells (any single time)
        direction: "positive" (heat/wet), "negative" (cold/dry), or "both"
        max_events: Cap returned event count (largest by peak |z|)

    Returns:
        Event catalogue: list of {time_start, time_end, duration_days, peak_sigma,
        peak_lat, peak_lon, footprint_cells, sign}
    """
    _ensure_materialized(dataset_id)
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    if direction not in ("positive", "negative", "both"):
        return {"error": "direction must be 'positive', 'negative', or 'both'"}

    try:
        from scipy import ndimage
    except ImportError:
        return {"error": "scipy is required for event detection"}

    try:
        data = _get_main_array(ds, metadata)
        if "time" not in data.dims:
            return {"error": "Dataset has no time dimension"}

        _progress(1, 4, "Materializing anomaly field...")
        arr = data.compute().values  # (time, lat, lon)
        times = data.time.values
        lats = data.lat.values
        lons = data.lon.values

        results = []
        signs = []
        if direction in ("positive", "both"):
            signs.append(("positive", +1))
        if direction in ("negative", "both"):
            signs.append(("negative", -1))

        for sign_label, sign in signs:
            _progress(2, 4, f"Labeling {sign_label} extremes...")
            mask = (sign * arr) >= sigma_threshold
            mask = np.where(np.isnan(arr), False, mask)
            if not mask.any():
                continue

            # 3D connectivity: time + spatial neighbours
            labels, n_events = ndimage.label(mask)
            if n_events == 0:
                continue

            _progress(3, 4, f"Cataloguing {n_events} {sign_label} events...")
            for ev_id in range(1, n_events + 1):
                idx = np.where(labels == ev_id)
                t_idx, y_idx, x_idx = idx
                duration = int(np.unique(t_idx).size)
                if duration < min_duration_days:
                    continue
                # max footprint across time slices
                footprint = max(
                    int(((labels[t] == ev_id).sum())) for t in np.unique(t_idx)
                )
                if footprint < min_area_cells:
                    continue
                vals = arr[t_idx, y_idx, x_idx]
                peak_local = int(np.argmax(sign * vals))
                peak_t = int(t_idx[peak_local])
                peak_y = int(y_idx[peak_local])
                peak_x = int(x_idx[peak_local])
                results.append({
                    "sign": sign_label,
                    "time_start": str(times[int(t_idx.min())])[:10],
                    "time_end": str(times[int(t_idx.max())])[:10],
                    "duration_days": duration,
                    "footprint_cells": footprint,
                    "peak_sigma": round(float(vals[peak_local]), 3),
                    "peak_time": str(times[peak_t])[:10],
                    "peak_lat": round(float(lats[peak_y]), 3),
                    "peak_lon": round(float(lons[peak_x]), 3),
                })

        _progress(4, 4, f"Found {len(results)} events")
        # Rank by absolute peak sigma
        results.sort(key=lambda e: abs(e["peak_sigma"]), reverse=True)
        truncated = len(results) > max_events
        results = results[:max_events]

    except Exception as e:
        logger.exception("detect_extreme_events failed", extra={"tool": "detect_extreme_events"})
        return {"error": f"Failed to detect events: {str(e)}"}

    return {
        "success": True,
        "dataset_id": dataset_id,
        "event_count": len(results),
        "truncated": truncated,
        "sigma_threshold": sigma_threshold,
        "direction": direction,
        "events": results,
    }


@mcp.tool()
def calculate_eof(
    dataset_id: str,
    n_modes: int = 3,
    detrend: bool = True,
) -> dict:
    """
    Empirical Orthogonal Function (EOF / PCA) decomposition of a 3D field.

    Returns the leading spatial patterns of variability and their principal-component
    time series. Standard climate diagnostic for finding dominant modes (e.g.,
    teleconnection-like patterns). Apply cosine-latitude weighting before SVD.

    For best signal, run on anomalies (calculate_anomaly or calculate_standardized_anomaly)
    and on a coarsened/subsetted field — full-globe daily input is RAM-heavy.

    Args:
        dataset_id: ID of a 3D (time, lat, lon) dataset
        n_modes: Number of leading EOFs to return (default 3, max 10)
        detrend: Remove linear trend per grid cell before decomposition (default True)

    Returns:
        spatial_dataset_id (lat, lon, mode), pc_dataset_id (time, mode),
        variance_explained per mode (fraction)
    """
    _ensure_materialized(dataset_id)
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    n_modes = max(1, min(int(n_modes), 10))

    try:
        data = _get_main_array(ds, metadata)
        if not all(d in data.dims for d in ("time", "lat", "lon")):
            return {"error": "EOF requires (time, lat, lon) dimensions"}

        _progress(1, 5, "Materializing field...")
        data = data.transpose("time", "lat", "lon").compute()

        if detrend:
            _progress(2, 5, "Detrending per grid cell...")
            t_idx = np.arange(data.sizes["time"])
            # Linear detrend along time axis
            from scipy import signal
            arr = data.values.astype(np.float64)
            mask_finite = np.isfinite(arr)
            # signal.detrend can't handle NaNs — fill, detrend, restore
            arr_filled = np.where(mask_finite, arr, 0.0)
            detrended = signal.detrend(arr_filled, axis=0, type="linear")
            arr = np.where(mask_finite, detrended, np.nan)
        else:
            arr = data.values.astype(np.float64)
            # Centre on time mean
            arr = arr - np.nanmean(arr, axis=0, keepdims=True)

        _progress(3, 5, "Applying cos(lat) weights...")
        lat_w = np.sqrt(np.cos(np.deg2rad(data.lat.values))).clip(0)
        weighted = arr * lat_w[np.newaxis, :, np.newaxis]

        n_time, n_lat, n_lon = weighted.shape
        flat = weighted.reshape(n_time, n_lat * n_lon)
        # Drop spatial columns that are all-NaN (e.g., land mask)
        col_finite = np.isfinite(flat).all(axis=0)
        flat_clean = np.nan_to_num(flat[:, col_finite], nan=0.0)
        if flat_clean.shape[1] < n_modes:
            return {"error": "Too few valid grid cells for the requested number of modes"}

        _progress(4, 5, f"SVD ({n_time}×{flat_clean.shape[1]})...")
        # Truncated SVD via numpy (safe for moderate fields). For huge fields we'd
        # want sklearn TruncatedSVD or dask-ml; document upstream.
        U, S, Vt = np.linalg.svd(flat_clean, full_matrices=False)
        var_frac = (S ** 2) / np.sum(S ** 2)

        # Spatial modes: rebuild full grid with NaNs restored
        spatial_full = np.full((n_modes, n_lat * n_lon), np.nan)
        spatial_full[:, col_finite] = Vt[:n_modes, :]
        # Remove cos-lat weighting from displayed pattern
        spatial_full = spatial_full.reshape(n_modes, n_lat, n_lon)
        with np.errstate(invalid="ignore", divide="ignore"):
            spatial_full = spatial_full / lat_w[np.newaxis, :, np.newaxis]

        # PCs scaled by singular values for proper amplitude
        pcs = U[:, :n_modes] * S[np.newaxis, :n_modes]

        _progress(5, 5, "Storing modes and PCs...")
        spatial_da = xr.DataArray(
            spatial_full,
            dims=["mode", "lat", "lon"],
            coords={"mode": np.arange(1, n_modes + 1), "lat": data.lat, "lon": data.lon},
            name=f"{metadata.variable}_eof",
        )
        pc_da = xr.DataArray(
            pcs,
            dims=["time", "mode"],
            coords={"time": data.time, "mode": np.arange(1, n_modes + 1)},
            name=f"{metadata.variable}_pc",
        )

        spatial_id = session_manager.store(
            data=spatial_da,
            source="computed",
            variable=f"{metadata.variable}_eof",
            model=metadata.model,
            scenario=metadata.scenario,
            description=f"EOF spatial modes 1-{n_modes} of {dataset_id}",
        )
        pc_id = session_manager.store(
            data=pc_da,
            source="computed",
            variable=f"{metadata.variable}_pc",
            model=metadata.model,
            scenario=metadata.scenario,
            description=f"EOF principal components 1-{n_modes} of {dataset_id}",
        )

    except Exception as e:
        logger.exception("calculate_eof failed", extra={"tool": "calculate_eof"})
        return {"error": f"Failed to compute EOFs: {str(e)}"}

    return {
        "success": True,
        "spatial_dataset_id": spatial_id,
        "pc_dataset_id": pc_id,
        "n_modes": n_modes,
        "variance_explained": [round(float(v), 4) for v in var_frac[:n_modes]],
        "cumulative_variance_explained": round(float(np.sum(var_frac[:n_modes])), 4),
        "note": (
            f"Plot spatial modes with generate_map(dataset_id='{spatial_id}') "
            f"and PC time series with generate_timeseries_plot(['{pc_id}'])."
        ),
    }


# ============================================================================
# Pillar 2 — Multi-model ensembles & scenario comparison
# ============================================================================

_MAX_ENSEMBLE_MODELS = 10
_MAX_SCENARIOS_AT_ONCE = 5


@mcp.tool()
def load_multi_model_ensemble(
    variable: str,
    models: list[str],
    scenario: str,
    start_date: str,
    end_date: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> dict:
    """
    Load the same variable/scenario/region across multiple CMIP6 models in one call.

    Stacks results on a new `model` dimension so downstream tools can compute
    ensemble statistics. For RAM safety, capped at 10 models — pre-subset by region
    and time before invoking.

    Args:
        variable: Climate variable (tasmax, pr, etc.)
        models: List of CMIP6 model names (max 10)
        scenario: Emissions scenario (historical, ssp126, ssp245, ssp370, ssp585)
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        lat_min, lat_max, lon_min, lon_max: Bounding box

    Returns:
        ensemble_dataset_id, list of contributing models, dimensions
    """
    from rcmes_mcp.tools.data_access import load_climate_data

    if not models:
        return {"error": "Provide at least one model"}
    if len(models) > _MAX_ENSEMBLE_MODELS:
        return {"error": f"Too many models ({len(models)}). Cap is {_MAX_ENSEMBLE_MODELS} — pre-select."}

    loaded_arrays = []
    loaded_models = []
    failures = []
    total = len(models)
    for i, m in enumerate(models, start=1):
        _progress(i, total + 1, f"Loading {m} ({i}/{total})...")
        result = load_climate_data(
            variable=variable, model=m, scenario=scenario,
            start_date=start_date, end_date=end_date,
            lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
        )
        if not result.get("success"):
            failures.append({"model": m, "error": result.get("error", "unknown")})
            continue
        ds_id = result["dataset_id"]
        # Wait for any background materialization
        from rcmes_mcp.tools.data_access import wait_for_materialization
        wait_for_materialization(ds_id, timeout=600)
        ds = session_manager.get(ds_id)
        meta = session_manager.get_metadata(ds_id)
        if isinstance(ds, xr.Dataset):
            var_name = variable if variable in ds.data_vars else list(ds.data_vars)[0]
            arr = ds[var_name]
        else:
            arr = ds
        loaded_arrays.append(arr)
        loaded_models.append(m)

    if not loaded_arrays:
        return {"error": "All models failed to load", "failures": failures}

    _progress(total + 1, total + 1, "Stacking ensemble...")
    try:
        # Align time/space across models. NEX-GDDP shares the same grid, so concat is safe.
        ensemble = xr.concat(loaded_arrays, dim=xr.Variable("model", loaded_models))
    except Exception as e:
        return {"error": f"Failed to stack ensemble: {str(e)}", "loaded_models": loaded_models}

    new_id = session_manager.store(
        data=ensemble,
        source="NEX-GDDP-CMIP6",
        variable=variable,
        model=",".join(loaded_models),
        scenario=scenario,
        description=f"Ensemble of {len(loaded_models)} models for {variable}/{scenario} {start_date}–{end_date}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "models_loaded": loaded_models,
        "models_failed": failures,
        "scenario": scenario,
        "variable": variable,
        "dimensions": dict(ensemble.sizes),
        "note": "Run calculate_ensemble_statistics or generate_ensemble_spread_plot next.",
    }


@mcp.tool()
def calculate_ensemble_statistics(
    ensemble_dataset_id: str,
    baseline_start: str | None = None,
    baseline_end: str | None = None,
) -> dict:
    """
    Reduce a multi-model ensemble across the `model` dimension.

    Always returns ensemble mean, std, min, max as separate dataset_ids. If a
    baseline window is provided, also returns a model-agreement map: for each
    cell, the fraction of models agreeing with the ensemble mean on the SIGN of
    the change (full-period mean − baseline mean). This is the "stippling" used
    in IPCC figures to indicate confidence.

    Args:
        ensemble_dataset_id: Dataset with a `model` dim (from load_multi_model_ensemble)
        baseline_start: Optional baseline window start (YYYY-MM-DD) for agreement map
        baseline_end: Optional baseline window end

    Returns:
        Dict of dataset_ids: ensemble_mean_id, ensemble_std_id, ensemble_min_id,
        ensemble_max_id, and (if baseline given) agreement_id + n_models
    """
    _ensure_materialized(ensemble_dataset_id)
    try:
        ds = session_manager.get(ensemble_dataset_id)
        metadata = session_manager.get_metadata(ensemble_dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    if "model" not in ds.dims:
        return {"error": "Input is not a multi-model ensemble (missing 'model' dim)"}

    try:
        data = _get_main_array(ds, metadata) if isinstance(ds, xr.Dataset) else ds
        n_models = int(data.sizes["model"])

        _progress(1, 4, "Computing ensemble mean...")
        em = data.mean(dim="model")
        _progress(2, 4, "Computing ensemble spread (std)...")
        es = data.std(dim="model")
        _progress(3, 4, "Computing min/max envelope...")
        emn = data.min(dim="model")
        emx = data.max(dim="model")

        ids = {}
        ids["ensemble_mean_id"] = session_manager.store(
            data=em, source="computed", variable=f"{metadata.variable}_ensmean",
            model=metadata.model, scenario=metadata.scenario,
            description=f"Ensemble mean ({n_models} models) of {ensemble_dataset_id}",
        )
        ids["ensemble_std_id"] = session_manager.store(
            data=es, source="computed", variable=f"{metadata.variable}_ensstd",
            model=metadata.model, scenario=metadata.scenario,
            description=f"Ensemble std ({n_models} models) of {ensemble_dataset_id}",
        )
        ids["ensemble_min_id"] = session_manager.store(
            data=emn, source="computed", variable=f"{metadata.variable}_ensmin",
            model=metadata.model, scenario=metadata.scenario,
            description=f"Ensemble min ({n_models} models)",
        )
        ids["ensemble_max_id"] = session_manager.store(
            data=emx, source="computed", variable=f"{metadata.variable}_ensmax",
            model=metadata.model, scenario=metadata.scenario,
            description=f"Ensemble max ({n_models} models)",
        )

        agreement_block = {}
        if baseline_start and baseline_end:
            _progress(4, 4, "Computing model agreement vs baseline...")
            try:
                baseline_start, baseline_end = validate_date_range(baseline_start, baseline_end)
            except ValueError as e:
                return {"error": str(e)}

            # Per-model change = full-period mean - baseline mean
            full_mean = data.mean(dim="time")
            baseline_mean = data.sel(time=slice(baseline_start, baseline_end)).mean(dim="time")
            change = (full_mean - baseline_mean).compute()
            ensemble_change_sign = np.sign(change.mean(dim="model"))
            same_sign = (np.sign(change) == ensemble_change_sign)
            agreement = same_sign.sum(dim="model") / n_models
            agreement = agreement.rename(f"{metadata.variable}_agreement")
            agreement_id = session_manager.store(
                data=agreement, source="computed",
                variable=f"{metadata.variable}_agreement",
                model=metadata.model, scenario=metadata.scenario,
                description=(
                    f"Model agreement on sign of change vs baseline "
                    f"{baseline_start}–{baseline_end} ({n_models} models)"
                ),
            )
            agreement_block = {
                "agreement_id": agreement_id,
                "baseline_period": {"start": baseline_start, "end": baseline_end},
                "interpretation": "0.0 = full disagreement, 1.0 = all models agree on sign",
            }

    except Exception as e:
        logger.exception("calculate_ensemble_statistics failed", extra={"tool": "calculate_ensemble_statistics"})
        return {"error": f"Failed to compute ensemble statistics: {str(e)}"}

    return {
        "success": True,
        "n_models": n_models,
        **ids,
        **agreement_block,
    }


@mcp.tool()
def compare_scenarios(
    variable: str,
    model: str,
    scenarios: list[str],
    start_date: str,
    end_date: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> dict:
    """
    Load the same variable/model/region across multiple SSP scenarios.

    Returns a dataset_id per scenario plus dataset_ids for pairwise differences
    (each later − each earlier scenario). Use generate_scenario_fan_chart on the
    returned per-scenario IDs for the canonical comparison plot.

    Args:
        variable: Climate variable
        model: Climate model
        scenarios: List of scenario IDs (max 5), e.g. ["ssp126","ssp245","ssp585"]
        start_date, end_date: Time window (YYYY-MM-DD)
        lat_min..lon_max: Bounding box

    Returns:
        per_scenario: {scenario: dataset_id}, differences: list of {a, b, dataset_id}
    """
    from rcmes_mcp.tools.data_access import load_climate_data, wait_for_materialization

    if not scenarios:
        return {"error": "Provide at least one scenario"}
    if len(scenarios) > _MAX_SCENARIOS_AT_ONCE:
        return {"error": f"Too many scenarios ({len(scenarios)}); cap is {_MAX_SCENARIOS_AT_ONCE}"}

    per_scenario: dict[str, str] = {}
    failures = []
    for i, sc in enumerate(scenarios, start=1):
        _progress(i, len(scenarios) + 1, f"Loading scenario {sc}...")
        r = load_climate_data(
            variable=variable, model=model, scenario=sc,
            start_date=start_date, end_date=end_date,
            lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
        )
        if not r.get("success"):
            failures.append({"scenario": sc, "error": r.get("error", "unknown")})
            continue
        per_scenario[sc] = r["dataset_id"]
        wait_for_materialization(r["dataset_id"], timeout=600)

    if not per_scenario:
        return {"error": "All scenarios failed to load", "failures": failures}

    _progress(len(scenarios) + 1, len(scenarios) + 1, "Computing pairwise differences...")
    sc_keys = list(per_scenario.keys())
    differences = []
    try:
        for i in range(len(sc_keys)):
            for j in range(i + 1, len(sc_keys)):
                a, b = sc_keys[i], sc_keys[j]
                ds_a = session_manager.get(per_scenario[a])
                ds_b = session_manager.get(per_scenario[b])
                meta_a = session_manager.get_metadata(per_scenario[a])
                arr_a = _get_main_array(ds_a, meta_a) if isinstance(ds_a, xr.Dataset) else ds_a
                meta_b = session_manager.get_metadata(per_scenario[b])
                arr_b = _get_main_array(ds_b, meta_b) if isinstance(ds_b, xr.Dataset) else ds_b
                # Time-mean difference (cheap, scenario-comparable)
                diff = arr_b.mean(dim="time") - arr_a.mean(dim="time")
                diff_id = session_manager.store(
                    data=diff, source="computed",
                    variable=f"{variable}_diff_{b}_minus_{a}",
                    model=model, scenario=f"{b}-{a}",
                    description=f"Time-mean difference: {b} − {a} for {variable}/{model}",
                )
                differences.append({"a": a, "b": b, "dataset_id": diff_id})
    except Exception as e:
        logger.exception("compare_scenarios diff failed")
        return {"error": f"Failed pairwise diff: {str(e)}", "per_scenario": per_scenario}

    return {
        "success": True,
        "variable": variable,
        "model": model,
        "per_scenario": per_scenario,
        "differences": differences,
        "failures": failures,
        "note": (
            "Use generate_scenario_fan_chart with these per_scenario ids, "
            "or generate_map on a difference id to see the spatial change."
        ),
    }


@mcp.tool()
def calculate_time_of_emergence(
    dataset_id: str,
    baseline_start: str,
    baseline_end: str,
    rolling_years: int = 20,
    sigma_threshold: float = 1.0,
) -> dict:
    """
    Map the year at which the climate-change signal emerges from natural variability.

    For each grid cell:
      1. baseline mean μ₀ and std σ₀ are computed from the baseline window
      2. an annual-mean series is built and rolled over `rolling_years`
      3. the first year where |rolling_mean − μ₀| > sigma_threshold × σ₀ is recorded
      NaN where emergence never occurs in the dataset window.

    Args:
        dataset_id: 3D (time, lat, lon) dataset, typically a future projection
        baseline_start: YYYY-MM-DD
        baseline_end: YYYY-MM-DD
        rolling_years: Width of rolling window (default 20)
        sigma_threshold: Multiplier on baseline std (default 1.0; use 2 for stricter)

    Returns:
        emergence_dataset_id: 2D (lat, lon) of emergence year (or NaN)
    """
    _ensure_materialized(dataset_id)
    try:
        baseline_start, baseline_end = validate_date_range(baseline_start, baseline_end)
    except ValueError as e:
        return {"error": str(e)}
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    try:
        data = _get_main_array(ds, metadata) if isinstance(ds, xr.Dataset) else ds
        if not all(d in data.dims for d in ("time", "lat", "lon")):
            return {"error": "calculate_time_of_emergence requires (time, lat, lon)"}

        _progress(1, 4, "Annual mean & baseline stats...")
        annual = data.resample(time="1YS").mean().compute()
        baseline = annual.sel(time=slice(baseline_start, baseline_end))
        mu0 = baseline.mean(dim="time")
        sigma0 = baseline.std(dim="time")

        _progress(2, 4, f"Rolling {rolling_years}-year mean...")
        rolled = annual.rolling(time=rolling_years, center=True, min_periods=max(2, rolling_years // 2)).mean()

        _progress(3, 4, "Locating emergence year...")
        deviation = np.abs(rolled - mu0)
        threshold = sigma_threshold * sigma0
        emerged = deviation > threshold
        # First True along time → year value; if never True → NaN
        years = rolled["time"].dt.year
        # broadcast year along lat/lon
        years_b = xr.DataArray(years.values, dims=["time"], coords={"time": rolled.time})
        # Where emerged, take year; else NaN. Then min over time.
        yr_field = years_b.where(emerged)
        emergence_year = yr_field.min(dim="time", skipna=True)
        emergence_year.name = f"{metadata.variable}_emergence_year"

        _progress(4, 4, "Storing emergence map...")
        new_id = session_manager.store(
            data=emergence_year, source="computed",
            variable=f"{metadata.variable}_emergence_year",
            model=metadata.model, scenario=metadata.scenario,
            description=(
                f"Time-of-emergence year (|Δ| > {sigma_threshold}σ over {rolling_years}yr "
                f"window) vs baseline {baseline_start}–{baseline_end}"
            ),
        )

    except Exception as e:
        logger.exception("calculate_time_of_emergence failed", extra={"tool": "calculate_time_of_emergence"})
        return {"error": f"Failed to compute time-of-emergence: {str(e)}"}

    return {
        "success": True,
        "dataset_id": new_id,
        "baseline_period": {"start": baseline_start, "end": baseline_end},
        "rolling_years": rolling_years,
        "sigma_threshold": sigma_threshold,
        "note": "Plot with generate_map (NaN cells never emerge in the input window).",
    }

