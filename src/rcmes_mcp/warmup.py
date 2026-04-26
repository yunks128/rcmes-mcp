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

from rcmes_mcp.utils.cloud import (
    CMIP6_MODELS,
    _get_cached_kerchunk_ref,
    open_nex_gddp_dataset,
)

logger = logging.getLogger("rcmes.warmup")

# Pre-defined regions — matches the chat UI guided selection menu
REGIONS = {
    "california": {
        "name": "California",
        "lat_min": 32.0, "lat_max": 42.0,
        "lon_min": -124.0, "lon_max": -114.0,
    },
    "texas": {
        "name": "Texas",
        "lat_min": 25.5, "lat_max": 36.5,
        "lon_min": -106.5, "lon_max": -93.5,
    },
    "thailand": {
        "name": "Thailand",
        "lat_min": 5.5, "lat_max": 20.5,
        "lon_min": 97.0, "lon_max": 106.0,
    },
    "florida": {
        "name": "Florida",
        "lat_min": 24.5, "lat_max": 31.0,
        "lon_min": -87.5, "lon_max": -80.0,
    },
    "conus": {
        "name": "Continental US",
        "lat_min": 24.0, "lat_max": 50.0,
        "lon_min": -125.0, "lon_max": -66.0,
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

# ============================================================================
# Default warmup config — matches the chat UI guided selection menus exactly
# ============================================================================

# The 5 models suggested in the chat flow
DEFAULT_MODELS = ["ACCESS-CM2", "GFDL-ESM4", "MRI-ESM2-0", "CanESM5", "CESM2"]

# All 4 future scenarios + historical
DEFAULT_SCENARIOS = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]

# The 4 variables suggested in the chat flow
DEFAULT_VARIABLES = ["tasmax", "tasmin", "pr", "tas"]

# The 3 regions most commonly selected (California, Texas, Thailand)
DEFAULT_REGIONS = ["california", "texas", "thailand"]

# Time ranges split into decades for manageable download sizes (~30s each)
DEFAULT_YEAR_RANGES = {
    "historical": [(2000, 2009), (2010, 2014)],
    "ssp126": [(2020, 2029), (2030, 2039), (2040, 2049), (2050, 2059)],
    "ssp245": [(2020, 2029), (2030, 2039), (2040, 2049), (2050, 2059)],
    "ssp370": [(2020, 2029), (2030, 2039), (2040, 2049), (2050, 2059)],
    "ssp585": [(2020, 2029), (2030, 2039), (2040, 2049), (2050, 2059)],
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


def _download_one(combo, year_ranges, dry_run=False):
    """Download and cache a single subset. Returns (label, status, elapsed)."""
    from rcmes_mcp.tools.data_access import _cache_subset, _get_cached_subset, _get_subset_cache_key

    region_key, region, model, scenario, variable, (start_yr, end_yr) = combo
    label = f"{variable}/{model}/{scenario} @ {region['name']} ({start_yr}-{end_yr})"

    lon_min_360 = region["lon_min"] if region["lon_min"] >= 0 else 360 + region["lon_min"]
    lon_max_360 = region["lon_max"] if region["lon_max"] >= 0 else 360 + region["lon_max"]

    cache_key = _get_subset_cache_key(
        variable, model, scenario,
        f"{start_yr}-01-01", f"{end_yr}-12-31",
        region["lat_min"], region["lat_max"],
        lon_min_360, lon_max_360,
    )

    if _get_cached_subset(cache_key) is not None:
        return (label, "cached", 0)

    if dry_run:
        return (label, "dry-run", 0)

    t0 = time.perf_counter()
    try:
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
        ds = ds.compute()
        _cache_subset(cache_key, ds)
        elapsed = time.perf_counter() - t0
        return (label, "done", elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return (label, f"FAILED: {e}", elapsed)


def warmup_subsets(
    models: list[str],
    scenarios: list[str],
    variables: list[str],
    regions: dict[str, dict],
    year_ranges: dict[str, list[tuple[int, int]]],
    dry_run: bool = False,
    parallel: int = 1,
) -> int:
    """Pre-download and cache data subsets.

    Returns number of subsets cached.
    """
    pass  # imports are inline below for parallel mode

    combos = []
    for region_key, region in regions.items():
        for model in models:
            for scenario in scenarios:
                for variable in variables:
                    for yr_range in year_ranges.get(scenario, [(2000, 2014)]):
                        combos.append((region_key, region, model, scenario, variable, yr_range))

    total = len(combos)
    cached = 0
    skipped = 0
    failed = 0

    def _print_result(idx, result_label, status, elapsed):
        nonlocal cached, skipped, failed
        if status == "cached":
            print(f"  [{idx}/{total}] (cached) {result_label}")
            skipped += 1
        elif status == "dry-run":
            print(f"  [{idx}/{total}] (dry-run) {result_label}")
        elif status == "done":
            print(f"  [{idx}/{total}] done ({elapsed:.1f}s) {result_label}")
            cached += 1
        else:
            print(f"  [{idx}/{total}] {status} — {result_label}")
            failed += 1

    if parallel <= 1:
        for i, combo in enumerate(combos):
            result_label, status, elapsed = _download_one(combo, year_ranges, dry_run)
            _print_result(i + 1, result_label, status, elapsed)
    else:
        # Use ProcessPoolExecutor — Dask is not thread-safe for concurrent compute()
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"  Using {parallel} parallel processes")
        done_count = 0
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_download_one, combo, year_ranges, dry_run): i
                for i, combo in enumerate(combos)
            }
            for future in as_completed(futures):
                done_count += 1
                result_label, status, elapsed = future.result()
                _print_result(done_count, result_label, status, elapsed)

    if failed:
        print(f"\n  Warning: {failed} downloads failed")
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
        "--parallel", "-j", type=int, default=1,
        help="Number of parallel downloads (default: 1, max recommended: 4)",
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
        DEFAULT_YEAR_RANGES, dry_run=args.dry_run,
        parallel=args.parallel,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n    Cached {cached_count} new subsets in {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
