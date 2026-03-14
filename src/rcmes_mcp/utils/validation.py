"""
Shared input validation for RCMES-MCP tools and API endpoints.

All validators raise ValueError with human-readable messages.
Callers convert to their own error format ({"error": ...} for MCP tools,
HTTPException for API).
"""

from __future__ import annotations

from datetime import datetime

import xarray as xr

# --- Constants ---

DATE_FORMAT = "%Y-%m-%d"

# NEX-GDDP-CMIP6 spatial coverage
LAT_MIN, LAT_MAX = -60.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0

# Guard rails
MAX_TIME_SPAN_DAYS = 36525  # ~100 years
MAX_DOWNLOAD_ELEMENTS = 500_000_000  # 500M elements ≈ 2 GB float32


def validate_date(date_str: str) -> str:
    """Validate and normalize a date string to YYYY-MM-DD format.

    Returns the canonical date string.
    Raises ValueError if the format is invalid.
    """
    try:
        dt = datetime.strptime(date_str, DATE_FORMAT)
    except (ValueError, TypeError):
        raise ValueError(
            f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD (e.g. 2050-01-01)."
        )
    return dt.strftime(DATE_FORMAT)


def validate_date_range(
    start_date: str,
    end_date: str,
    max_span_days: int = MAX_TIME_SPAN_DAYS,
) -> tuple[str, str]:
    """Validate a date range: format, ordering, and maximum span.

    Returns (start_date, end_date) as canonical strings.
    Raises ValueError on any issue.
    """
    start_date = validate_date(start_date)
    end_date = validate_date(end_date)

    start_dt = datetime.strptime(start_date, DATE_FORMAT)
    end_dt = datetime.strptime(end_date, DATE_FORMAT)

    if start_dt >= end_dt:
        raise ValueError(
            f"start_date ({start_date}) must be before end_date ({end_date})."
        )

    span = (end_dt - start_dt).days
    if span > max_span_days:
        raise ValueError(
            f"Date range spans {span} days, exceeding the maximum of {max_span_days} days "
            f"(~{max_span_days // 365} years). Use a shorter range."
        )

    return start_date, end_date


def validate_lat_lon_bounds(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> None:
    """Validate geographic bounding box coordinates.

    Raises ValueError if coordinates are out of range or inverted.
    """
    if not (LAT_MIN <= lat_min <= LAT_MAX):
        raise ValueError(
            f"lat_min ({lat_min}) out of range [{LAT_MIN}, {LAT_MAX}]."
        )
    if not (LAT_MIN <= lat_max <= LAT_MAX):
        raise ValueError(
            f"lat_max ({lat_max}) out of range [{LAT_MIN}, {LAT_MAX}]."
        )
    if lat_min >= lat_max:
        raise ValueError(
            f"lat_min ({lat_min}) must be less than lat_max ({lat_max})."
        )
    if not (LON_MIN <= lon_min <= LON_MAX):
        raise ValueError(
            f"lon_min ({lon_min}) out of range [{LON_MIN}, {LON_MAX}]."
        )
    if not (LON_MIN <= lon_max <= LON_MAX):
        raise ValueError(
            f"lon_max ({lon_max}) out of range [{LON_MIN}, {LON_MAX}]."
        )
    if lon_min >= lon_max:
        raise ValueError(
            f"lon_min ({lon_min}) must be less than lon_max ({lon_max})."
        )


def check_download_size(
    ds: xr.Dataset | xr.DataArray,
    max_elements: int = MAX_DOWNLOAD_ELEMENTS,
) -> None:
    """Check that a dataset is small enough to materialize for download.

    Raises ValueError if the total number of elements exceeds the limit.
    """
    if isinstance(ds, xr.DataArray):
        total = ds.size
    else:
        total = sum(ds[v].size for v in ds.data_vars)

    if total > max_elements:
        est_gb = total * 4 / (1024**3)  # float32 estimate
        max_gb = max_elements * 4 / (1024**3)
        raise ValueError(
            f"Dataset too large to download: ~{est_gb:.1f} GB estimated "
            f"({total:,} elements). Maximum is ~{max_gb:.1f} GB "
            f"({max_elements:,} elements). Apply spatial/temporal subsetting first."
        )
