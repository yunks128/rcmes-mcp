"""
Analysis Tools

MCP tools for climate data analysis including statistics, trends,
and model evaluation metrics.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats

from rcmes_mcp.server import mcp
from rcmes_mcp.utils.session import session_manager


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
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get the main data variable
    if isinstance(ds, xr.DataArray):
        data = ds
    else:
        var_name = metadata.variable or list(ds.data_vars)[0]
        data = ds[var_name]

    results = {
        "dataset_id": dataset_id,
        "variable": metadata.variable,
    }

    try:
        # Compute in chunks to avoid memory issues
        if statistic in ["mean", "all"]:
            results["mean"] = float(data.mean().compute())

        if statistic in ["std", "all"]:
            results["std"] = float(data.std().compute())

        if statistic in ["min", "all"]:
            results["min"] = float(data.min().compute())

        if statistic in ["max", "all"]:
            results["max"] = float(data.max().compute())

        if statistic in ["percentiles", "all"]:
            # Compute percentiles (subsample for large datasets)
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
        return {"error": f"Failed to compute statistics: {str(e)}"}

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
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    try:
        if period == "daily":
            climatology = ds.groupby("time.dayofyear").mean()
        elif period == "monthly":
            climatology = ds.groupby("time.month").mean()
        elif period == "seasonal":
            climatology = ds.groupby("time.season").mean()
        else:
            return {"error": f"Invalid period '{period}'. Use: daily, monthly, seasonal"}

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
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get the main data variable
    if isinstance(ds, xr.DataArray):
        data = ds
    else:
        var_name = metadata.variable or list(ds.data_vars)[0]
        data = ds[var_name]

    try:
        # Calculate area-weighted spatial mean time series
        weights = np.cos(np.deg2rad(data.lat))
        weights = weights / weights.sum()

        if "lat" in data.dims and "lon" in data.dims:
            time_series = data.weighted(weights).mean(dim=["lat", "lon"])
        else:
            time_series = data.mean(dim=[d for d in data.dims if d != "time"])

        # Convert to numpy
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

        # Convert slope to per-decade
        # Assuming daily data, 365.25 days per year
        days_per_decade = 365.25 * 10
        slope_per_decade = slope * days_per_decade

        # Confidence interval (95%)
        ci_95 = 1.96 * std_err * days_per_decade

    except Exception as e:
        return {"error": f"Failed to calculate trend: {str(e)}"}

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
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get the main data variable
    if isinstance(ds, xr.DataArray):
        data = ds
    else:
        var_name = metadata.variable or list(ds.data_vars)[0]
        data = ds[var_name]

    try:
        if area_weighted and "lat" in data.dims:
            # Calculate weights based on latitude
            weights = np.cos(np.deg2rad(data.lat))
            regional_mean = data.weighted(weights).mean(dim=["lat", "lon"])
        else:
            regional_mean = data.mean(dim=["lat", "lon"])

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

    # Calculate summary stats
    mean_val = float(regional_mean.mean().compute())
    std_val = float(regional_mean.std().compute())

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "area_weighted": area_weighted,
        "summary": {
            "mean": mean_val,
            "std": std_val,
            "time_steps": regional_mean.dims.get("time", len(regional_mean)),
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

        # Calculate overall statistics
        spatial_mean_bias = float(bias.mean().compute())
        spatial_rmse = float(np.sqrt(((model_mean - ref_mean) ** 2).mean()).compute())

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
