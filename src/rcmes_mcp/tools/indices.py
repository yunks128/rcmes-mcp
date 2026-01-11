"""
Climate Indices Tools

MCP tools for calculating ETCCDI climate extreme indices and other
climate indicators using xclim.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from rcmes_mcp.server import mcp
from rcmes_mcp.utils.session import session_manager

# ETCCDI indices documentation
TEMPERATURE_INDICES = {
    "TXx": "Maximum of daily maximum temperature",
    "TXn": "Minimum of daily maximum temperature",
    "TNx": "Maximum of daily minimum temperature",
    "TNn": "Minimum of daily minimum temperature",
    "TX90p": "Percentage of days when TX > 90th percentile (warm days)",
    "TX10p": "Percentage of days when TX < 10th percentile (cool days)",
    "TN90p": "Percentage of days when TN > 90th percentile (warm nights)",
    "TN10p": "Percentage of days when TN < 10th percentile (cold nights)",
    "WSDI": "Warm spell duration index",
    "CSDI": "Cold spell duration index",
    "DTR": "Diurnal temperature range",
    "SU": "Summer days (TX > 25°C)",
    "TR": "Tropical nights (TN > 20°C)",
    "FD": "Frost days (TN < 0°C)",
    "ID": "Ice days (TX < 0°C)",
    "GSL": "Growing season length",
}

PRECIPITATION_INDICES = {
    "Rx1day": "Maximum 1-day precipitation",
    "Rx5day": "Maximum 5-day precipitation",
    "SDII": "Simple daily intensity index",
    "R10mm": "Heavy precipitation days (precip >= 10mm)",
    "R20mm": "Very heavy precipitation days (precip >= 20mm)",
    "CDD": "Consecutive dry days",
    "CWD": "Consecutive wet days",
    "R95p": "Very wet day precipitation",
    "R99p": "Extremely wet day precipitation",
    "PRCPTOT": "Annual total wet-day precipitation",
}


@mcp.tool()
def list_climate_indices() -> dict:
    """
    List available ETCCDI climate extreme indices.

    Returns:
        Dictionary with temperature and precipitation indices
    """
    return {
        "temperature_indices": TEMPERATURE_INDICES,
        "precipitation_indices": PRECIPITATION_INDICES,
        "note": "Use calculate_etccdi_index to compute these indices from daily data",
    }


@mcp.tool()
def calculate_etccdi_index(
    dataset_id: str,
    index: str,
    freq: str = "YS",
    threshold: float | None = None,
) -> dict:
    """
    Calculate an ETCCDI climate extreme index.

    Args:
        dataset_id: ID of the dataset (must be daily temperature or precipitation)
        index: Index name (TXx, TX90p, Rx1day, CDD, etc.)
        freq: Output frequency - "YS" (annual), "QS-DEC" (seasonal), "MS" (monthly)
        threshold: Optional custom threshold (e.g., for SU, default 25°C)

    Returns:
        New dataset_id containing the calculated index

    Examples:
        # Calculate annual maximum temperature
        calculate_etccdi_index(dataset_id="ds_abc123", index="TXx", freq="YS")

        # Calculate warm days percentage
        calculate_etccdi_index(dataset_id="ds_abc123", index="TX90p", freq="YS")

        # Calculate maximum 5-day precipitation
        calculate_etccdi_index(dataset_id="ds_abc123", index="Rx5day", freq="YS")
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Check if xclim is available
    try:
        import xclim
        from xclim import indices
    except ImportError:
        return {"error": "xclim package not installed. Install with: pip install xclim"}

    # Get the data variable
    if isinstance(ds, xr.DataArray):
        data = ds
    else:
        var_name = metadata.variable or list(ds.data_vars)[0]
        data = ds[var_name]

    # Ensure proper units for xclim
    if "units" not in data.attrs:
        # Infer units from variable name
        if metadata.variable in ["tas", "tasmax", "tasmin"]:
            data.attrs["units"] = "K"
        elif metadata.variable == "pr":
            data.attrs["units"] = "kg m-2 s-1"

    index_upper = index.upper()

    try:
        # Temperature extreme indices
        if index_upper == "TXX":
            result = indices.tx_max(data, freq=freq)

        elif index_upper == "TXN":
            result = indices.tx_min(data, freq=freq)

        elif index_upper == "TNX":
            result = indices.tn_max(data, freq=freq)

        elif index_upper == "TNN":
            result = indices.tn_min(data, freq=freq)

        elif index_upper == "TX90P":
            # Need to calculate 90th percentile from baseline
            t90 = data.quantile(0.9, dim="time")
            result = indices.tx90p(data, t90, freq=freq)

        elif index_upper == "TX10P":
            t10 = data.quantile(0.1, dim="time")
            result = indices.tx10p(data, t10, freq=freq)

        elif index_upper == "TN90P":
            t90 = data.quantile(0.9, dim="time")
            result = indices.tn90p(data, t90, freq=freq)

        elif index_upper == "TN10P":
            t10 = data.quantile(0.1, dim="time")
            result = indices.tn10p(data, t10, freq=freq)

        elif index_upper == "DTR":
            # DTR requires both tasmax and tasmin
            return {"error": "DTR requires both tasmax and tasmin datasets. Use calculate_dtr instead."}

        elif index_upper == "SU":
            thresh = f"{threshold or 25} degC"
            result = indices.tx_days_above(data, thresh=thresh, freq=freq)

        elif index_upper == "TR":
            thresh = f"{threshold or 20} degC"
            result = indices.tn_days_above(data, thresh=thresh, freq=freq)

        elif index_upper == "FD":
            result = indices.frost_days(data, freq=freq)

        elif index_upper == "ID":
            result = indices.ice_days(data, freq=freq)

        elif index_upper == "GSL":
            result = indices.growing_season_length(data, freq=freq)

        # Precipitation extreme indices
        elif index_upper == "RX1DAY":
            result = indices.max_1day_precipitation_amount(data, freq=freq)

        elif index_upper == "RX5DAY":
            result = indices.max_n_day_precipitation_amount(data, window=5, freq=freq)

        elif index_upper == "SDII":
            result = indices.daily_pr_intensity(data, freq=freq)

        elif index_upper == "R10MM":
            result = indices.wetdays(data, thresh="10 mm/day", freq=freq)

        elif index_upper == "R20MM":
            result = indices.wetdays(data, thresh="20 mm/day", freq=freq)

        elif index_upper == "CDD":
            result = indices.maximum_consecutive_dry_days(data, freq=freq)

        elif index_upper == "CWD":
            result = indices.maximum_consecutive_wet_days(data, freq=freq)

        elif index_upper == "R95P":
            pr95 = data.quantile(0.95, dim="time")
            result = indices.days_over_precip_thresh(data, pr95, freq=freq)

        elif index_upper == "R99P":
            pr99 = data.quantile(0.99, dim="time")
            result = indices.days_over_precip_thresh(data, pr99, freq=freq)

        elif index_upper == "PRCPTOT":
            result = indices.prcptot(data, freq=freq)

        else:
            all_indices = list(TEMPERATURE_INDICES.keys()) + list(PRECIPITATION_INDICES.keys())
            return {
                "error": f"Unknown index '{index}'.",
                "available_indices": all_indices,
            }

    except Exception as e:
        return {"error": f"Failed to calculate {index}: {str(e)}"}

    new_id = session_manager.store(
        data=result,
        source=metadata.source,
        variable=index_upper,
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"{index_upper} index calculated from {dataset_id}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "index": index_upper,
        "frequency": freq,
        "description": TEMPERATURE_INDICES.get(index_upper) or PRECIPITATION_INDICES.get(index_upper),
    }


@mcp.tool()
def analyze_heatwaves(
    dataset_id: str,
    threshold_percentile: float = 90,
    min_duration: int = 3,
    baseline_start: str | None = None,
    baseline_end: str | None = None,
) -> dict:
    """
    Analyze heatwave characteristics including frequency, duration, and intensity.

    A heatwave is defined as a period of at least min_duration consecutive days
    where daily maximum temperature exceeds the threshold_percentile.

    Args:
        dataset_id: ID of the tasmax dataset (daily maximum temperature)
        threshold_percentile: Percentile for heatwave threshold (default: 90)
        min_duration: Minimum consecutive days for a heatwave (default: 3)
        baseline_start: Start date for percentile calculation (optional)
        baseline_end: End date for percentile calculation (optional)

    Returns:
        Heatwave statistics and new dataset_id with annual heatwave metrics
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get the data variable
    if isinstance(ds, xr.DataArray):
        tasmax = ds
    else:
        var_name = metadata.variable or list(ds.data_vars)[0]
        tasmax = ds[var_name]

    try:
        # Calculate threshold
        if baseline_start and baseline_end:
            baseline = tasmax.sel(time=slice(baseline_start, baseline_end))
        else:
            baseline = tasmax

        threshold = baseline.quantile(threshold_percentile / 100, dim="time")

        # Identify hot days
        hot_days = tasmax > threshold

        # Find heatwave events (consecutive hot days >= min_duration)
        # This is a simplified approach - for production, use xclim's heat_wave_index
        try:
            from xclim import indices
            hw_freq = indices.heat_wave_frequency(
                tasmax,
                thresh=f"{threshold_percentile}th percentile",
                window=min_duration,
                freq="YS",
            )
            hw_max_length = indices.heat_wave_max_length(
                tasmax,
                thresh=f"{threshold_percentile}th percentile",
                window=min_duration,
                freq="YS",
            )
            hw_total_length = indices.heat_wave_total_length(
                tasmax,
                thresh=f"{threshold_percentile}th percentile",
                window=min_duration,
                freq="YS",
            )

            # Combine into dataset
            hw_stats = xr.Dataset({
                "heatwave_frequency": hw_freq,
                "heatwave_max_duration": hw_max_length,
                "heatwave_total_days": hw_total_length,
            })

        except Exception:
            # Fallback: simple counting approach
            hot_days_annual = hot_days.resample(time="YS").sum()

            hw_stats = xr.Dataset({
                "hot_days_count": hot_days_annual,
            })

        # Calculate overall statistics
        if "heatwave_frequency" in hw_stats:
            mean_freq = float(hw_stats["heatwave_frequency"].mean().compute())
            mean_duration = float(hw_stats["heatwave_max_duration"].mean().compute())
        else:
            mean_freq = None
            mean_duration = None

        mean_hot_days = float(hot_days.resample(time="YS").sum().mean().compute())

    except Exception as e:
        return {"error": f"Failed to analyze heatwaves: {str(e)}"}

    new_id = session_manager.store(
        data=hw_stats,
        source=metadata.source,
        variable="heatwave_metrics",
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"Heatwave analysis of {dataset_id}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "parameters": {
            "threshold_percentile": threshold_percentile,
            "min_duration": min_duration,
            "baseline": f"{baseline_start} to {baseline_end}" if baseline_start else "full period",
        },
        "summary": {
            "mean_annual_hot_days": mean_hot_days,
            "mean_annual_heatwave_frequency": mean_freq,
            "mean_heatwave_duration": mean_duration,
        },
    }


@mcp.tool()
def calculate_drought_index(
    precipitation_dataset_id: str,
    index: str = "SPI",
    scale: int = 3,
    temperature_dataset_id: str | None = None,
) -> dict:
    """
    Calculate drought indices (SPI or SPEI).

    Args:
        precipitation_dataset_id: ID of precipitation dataset
        index: "SPI" (Standardized Precipitation Index) or
               "SPEI" (Standardized Precipitation-Evapotranspiration Index)
        scale: Time scale in months (1, 3, 6, 12, etc.)
        temperature_dataset_id: Required for SPEI - temperature dataset for PET calculation

    Returns:
        New dataset_id with drought index values
    """
    try:
        pr_ds = session_manager.get(precipitation_dataset_id)
        pr_meta = session_manager.get_metadata(precipitation_dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get precipitation data
    if isinstance(pr_ds, xr.DataArray):
        pr = pr_ds
    else:
        var_name = pr_meta.variable or list(pr_ds.data_vars)[0]
        pr = pr_ds[var_name]

    try:
        # Check for climate-indices or xclim
        try:
            from xclim import indices

            # Resample to monthly if needed
            if pr.time.dt.dayofyear.values[1] != pr.time.dt.dayofyear.values[0]:
                pr_monthly = pr.resample(time="MS").sum()
            else:
                pr_monthly = pr

            if index.upper() == "SPI":
                result = indices.standardized_precipitation_index(
                    pr_monthly,
                    freq="MS",
                    window=scale,
                    dist="gamma",
                    method="APP",
                )

            elif index.upper() == "SPEI":
                if temperature_dataset_id is None:
                    return {"error": "SPEI requires temperature_dataset_id for PET calculation"}

                tas_ds = session_manager.get(temperature_dataset_id)
                if isinstance(tas_ds, xr.DataArray):
                    tas = tas_ds
                else:
                    tas = tas_ds[list(tas_ds.data_vars)[0]]

                # Resample temperature to monthly
                tas_monthly = tas.resample(time="MS").mean()

                result = indices.standardized_precipitation_evapotranspiration_index(
                    pr_monthly,
                    tas_monthly,
                    freq="MS",
                    window=scale,
                    dist="gamma",
                    method="APP",
                )
            else:
                return {"error": f"Unknown index '{index}'. Use 'SPI' or 'SPEI'."}

        except ImportError:
            return {"error": "xclim package required for drought indices. Install with: pip install xclim"}

    except Exception as e:
        return {"error": f"Failed to calculate {index}: {str(e)}"}

    new_id = session_manager.store(
        data=result,
        source=pr_meta.source,
        variable=f"{index.upper()}{scale}",
        model=pr_meta.model,
        scenario=pr_meta.scenario,
        description=f"{index.upper()}-{scale} drought index from {precipitation_dataset_id}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": precipitation_dataset_id,
        "index": index.upper(),
        "scale_months": scale,
        "interpretation": {
            ">2.0": "Extremely wet",
            "1.5 to 2.0": "Very wet",
            "1.0 to 1.5": "Moderately wet",
            "-1.0 to 1.0": "Near normal",
            "-1.5 to -1.0": "Moderately dry",
            "-2.0 to -1.5": "Severely dry",
            "<-2.0": "Extremely dry",
        },
    }


@mcp.tool()
def calculate_growing_degree_days(
    dataset_id: str,
    base_temperature: float = 10.0,
    upper_threshold: float | None = 30.0,
) -> dict:
    """
    Calculate Growing Degree Days (GDD) for agricultural applications.

    GDD = max(0, (Tavg - base_temperature))

    Args:
        dataset_id: ID of daily mean temperature dataset
        base_temperature: Base temperature in Celsius (default: 10°C)
        upper_threshold: Upper temperature threshold (default: 30°C)

    Returns:
        New dataset_id with cumulative GDD
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    # Get temperature data
    if isinstance(ds, xr.DataArray):
        tas = ds
    else:
        var_name = metadata.variable or list(ds.data_vars)[0]
        tas = ds[var_name]

    try:
        # Convert to Celsius if in Kelvin
        if tas.attrs.get("units") == "K":
            tas = tas - 273.15

        # Calculate daily GDD
        gdd_daily = tas - base_temperature
        gdd_daily = gdd_daily.clip(min=0)

        if upper_threshold is not None:
            gdd_daily = gdd_daily.clip(max=upper_threshold - base_temperature)

        # Calculate annual cumulative GDD
        gdd_annual = gdd_daily.resample(time="YS").sum()

    except Exception as e:
        return {"error": f"Failed to calculate GDD: {str(e)}"}

    new_id = session_manager.store(
        data=gdd_annual,
        source=metadata.source,
        variable="GDD",
        model=metadata.model,
        scenario=metadata.scenario,
        description=f"Growing Degree Days (base {base_temperature}°C) from {dataset_id}",
    )

    return {
        "success": True,
        "dataset_id": new_id,
        "original_dataset_id": dataset_id,
        "parameters": {
            "base_temperature": base_temperature,
            "upper_threshold": upper_threshold,
            "units": "degree-days",
        },
        "mean_annual_gdd": float(gdd_annual.mean().compute()),
    }
