"""
Data Warmup CLI

Pre-downloads and caches popular region/model/scenario combinations so that
subsequent queries are served from local disk instead of S3.

Usage:
    rcmes-warmup                    # Download default popular combos
    rcmes-warmup --regions conus    # Only CONUS region
    rcmes-warmup --models ACCESS-CM2 CESM2  # Only specific models
    rcmes-warmup --dry-run          # Show what would be downloaded
    rcmes-warmup --cache-refs-only  # Only cache Kerchunk reference files
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import xarray as xr

from rcmes_mcp.utils.cloud import (
    CMIP6_MODELS,
    CMIP6_SCENARIOS,
    CMIP6_VARIABLES,
    DEFAULT_CHUNKS,
    _get_cached_kerchunk_ref,
    open_nex_gddp_dataset,
)

logger = logging.getLogger("rcmes.warmup")

# Popular pre-defined regions
REGIONS = {
    "conus": {
        "name": "Continental US",
        "lat_min": 24.0, "lat_max": 50.0,
        "lon_min": -125.0, "lon_max": -66.0,
    },
    "california": {
        "name": "California",
        "lat_min": 32.0, "lat_max": 42.0,
        "lon_min": -124.5, "lon_max": -114.0,
    },
    "europe": {
        "name": "Europe",
        "lat_min": 35.0, "lat_max": 72.0,
        "lon_min": -12.0, "lon_max": 45.0,
    },
    "south_asia": {
        "name": "South Asia",
        "lat_min": 5.0, "lat_max": 40.0,
        "lon_min": 60.0, "lon_max": 100.0,
    },
    "east_africa": {
        "name": "East Africa",
        "lat_min": -12.0, "lat_max": 18.0,
        "lon_min": 28.0, "lon_max": 52.0,
    },
}

# Default warmup configuration — popular combos that cover common use cases
DEFAULT_MODELS = ["ACCESS-CM2", "CESM2", "GFDL-ESM4", "MPI-ESM1-2-HR", "UKESM1-0-LL"]
DEFAULT_SCENARIOS = ["historical", "ssp245", "ssp585"]
DEFAULT_VARIABLES = ["tas", "tasmax", "pr"]
DEFAULT_REGIONS = ["conus", "california"]
DEFAULT_YEARS = {
    "historical": (2000, 2014),
    "ssp245": (2040, 2060),
    "ssp585": (2040, 2060),
}


def warmup_kerchunk_refs(
    models: list[str],
    scenarios: list[str],
    dry_run: bool = False,
) -> int:
    """Cache Kerchunk reference files for model/scenario combinations.

    Returns number of refs cached.
    """
    count = 0
    total = len(models) * len(scenarios)
    for i, model in enumerate(models):
        for scenario in scenarios:
            count += 1
            label = f"[{count}/{total}] {model}/{scenario}"
            if dry_run:
                print(f"  (dry-run) Would cache refs: {label}")
                continue
            print(f"  Caching refs: {label} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                ref_path = _get_cached_kerchunk_ref(model, scenario)
                elapsed = time.perf_counter() - t0
                is_local = not ref_path.startswith("s3://")
                status = f"cached locally ({elapsed:.1f}s)" if is_local else "using S3 (cache failed)"
                print(status)
            except Exception as e:
                print(f"FAILED: {e}")
    return count


def warmup_subsets(
    models: list[str],
    scenarios: list[str],
    variables: list[str],
    regions: dict[str, dict],
    year_ranges: dict[str, tuple[int, int]],
    dry_run: bool = False,
) -> int:
    """Pre-download and cache data subsets.

    Returns number of subsets cached.
    """
    from rcmes_mcp.tools.data_access import _cache_subset, _get_subset_cache_key, _get_cached_subset

    combos = []
    for region_key, region in regions.items():
        for model in models:
            for scenario in scenarios:
                for variable in variables:
                    years = year_ranges.get(scenario, (2000, 2014))
                    combos.append((region_key, region, model, scenario, variable, years))

    total = len(combos)
    cached = 0
    skipped = 0

    for i, (region_key, region, model, scenario, variable, (start_yr, end_yr)) in enumerate(combos):
        label = f"[{i+1}/{total}] {variable}/{model}/{scenario} @ {region['name']} ({start_yr}-{end_yr})"

        # Convert longitudes to 0-360 for cache key
        lon_min_360 = region["lon_min"] if region["lon_min"] >= 0 else 360 + region["lon_min"]
        lon_max_360 = region["lon_max"] if region["lon_max"] >= 0 else 360 + region["lon_max"]

        cache_key = _get_subset_cache_key(
            variable, model, scenario,
            f"{start_yr}-01-01", f"{end_yr}-12-31",
            region["lat_min"], region["lat_max"],
            lon_min_360, lon_max_360,
        )

        # Skip if already cached
        if _get_cached_subset(cache_key) is not None:
            print(f"  (cached) {label}")
            skipped += 1
            continue

        if dry_run:
            print(f"  (dry-run) Would download: {label}")
            continue

        print(f"  Downloading: {label} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            # Handle wrap-around longitudes
            if lon_min_360 >= lon_max_360:
                lon_bounds = None
            else:
                lon_bounds = (lon_min_360, lon_max_360)

            ds = open_nex_gddp_dataset(
                variable=variable,
                model=model,
                scenario=scenario,
                start_year=start_yr,
                end_year=end_yr,
                lat_bounds=(region["lat_min"], region["lat_max"]),
                lon_bounds=lon_bounds,
            )
            ds = ds.sel(time=slice(f"{start_yr}-01-01", f"{end_yr}-12-31"))

            # Materialize and cache (with compression + stats pre-computation)
            _cache_subset(cache_key, ds)
            elapsed = time.perf_counter() - t0
            print(f"done ({elapsed:.1f}s)")
            cached += 1
        except Exception as e:
            print(f"FAILED: {e}")

    return cached


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download and cache climate data for fast access",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Models to warm up (default: {', '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--scenarios", nargs="*", default=None,
        help=f"Scenarios (default: {', '.join(DEFAULT_SCENARIOS)})",
    )
    parser.add_argument(
        "--variables", nargs="*", default=None,
        help=f"Variables (default: {', '.join(DEFAULT_VARIABLES)})",
    )
    parser.add_argument(
        "--regions", nargs="*", default=None,
        help=f"Region keys (default: {', '.join(DEFAULT_REGIONS)}). Available: {', '.join(REGIONS.keys())}",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Warm up ALL 35 CMIP6 models (takes a long time)",
    )
    parser.add_argument(
        "--cache-refs-only", action="store_true",
        help="Only cache Kerchunk reference files, skip data download",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    models = CMIP6_MODELS if args.all_models else (args.models or DEFAULT_MODELS)
    scenarios = args.scenarios or DEFAULT_SCENARIOS
    variables = args.variables or DEFAULT_VARIABLES
    region_keys = args.regions or DEFAULT_REGIONS

    # Validate region keys
    regions = {}
    for key in region_keys:
        if key not in REGIONS:
            print(f"Error: Unknown region '{key}'. Available: {', '.join(REGIONS.keys())}")
            sys.exit(1)
        regions[key] = REGIONS[key]

    # Validate models
    for m in models:
        if m not in CMIP6_MODELS:
            print(f"Error: Unknown model '{m}'.")
            sys.exit(1)

    print("=" * 60)
    print("RCMES Data Warmup")
    print("=" * 60)
    print(f"Models:    {', '.join(models)}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Variables: {', '.join(variables)}")
    region_labels = [f"{k} ({v['name']})" for k, v in regions.items()]
    print(f"Regions:   {', '.join(region_labels)}")
    if args.dry_run:
        print("MODE:      DRY RUN")
    print()

    # Step 1: Cache Kerchunk references
    print("--- Step 1: Caching Kerchunk reference files ---")
    t0 = time.perf_counter()
    ref_count = warmup_kerchunk_refs(models, scenarios, dry_run=args.dry_run)
    print(f"    Processed {ref_count} reference files in {time.perf_counter() - t0:.1f}s\n")

    if args.cache_refs_only:
        print("Done (--cache-refs-only).")
        return

    # Step 2: Download and cache subsets
    print("--- Step 2: Downloading and caching data subsets ---")
    t0 = time.perf_counter()
    cached_count = warmup_subsets(
        models, scenarios, variables, regions,
        DEFAULT_YEARS, dry_run=args.dry_run,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n    Cached {cached_count} new subsets in {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
