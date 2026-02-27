"""
Workflow Prompts

MCP prompt templates for common climate analysis workflows.
"""

from __future__ import annotations

from rcmes_mcp.server import mcp


@mcp.prompt()
def heatwave_analysis() -> str:
    """
    Prompt template for analyzing heatwave trends in a region.
    """
    return """
# Heatwave Analysis Workflow

To analyze heatwave trends for a specific region and scenario:

## Step 1: Load Daily Maximum Temperature Data
```
load_climate_data(
    variable="tasmax",
    model="ACCESS-CM2",  # or another model
    scenario="ssp585",   # high emissions scenario
    start_date="2015-01-01",
    end_date="2100-12-31",
    lat_min=..., lat_max=...,  # your region
    lon_min=..., lon_max=...
)
```

## Step 2: Convert Units to Celsius
```
convert_units(dataset_id="ds_xxx", target_unit="degC")
```

## Step 3: Analyze Heatwaves
```
analyze_heatwaves(
    dataset_id="ds_xxx",
    threshold_percentile=90,
    min_duration=3,
    baseline_start="2015-01-01",
    baseline_end="2044-12-31"
)
```

## Step 4: Calculate Trend
```
calculate_trend(dataset_id="ds_xxx")
```

## Step 5: Visualize Results
```
generate_timeseries_plot(
    dataset_ids=["ds_xxx"],
    title="Annual Heatwave Days in Region",
    show_trend=True
)
```

## Interpretation
- Compare trends across different scenarios (ssp126 vs ssp585)
- Compare different models for uncertainty assessment
- Look for acceleration in later periods (2050-2100)
"""


@mcp.prompt()
def model_evaluation() -> str:
    """
    Prompt template for evaluating climate model performance.
    """
    return """
# Climate Model Evaluation Workflow

To evaluate how well a climate model simulates historical observations:

## Step 1: Load Model Data (Historical Period)
```
load_climate_data(
    variable="tas",
    model="ACCESS-CM2",
    scenario="historical",
    start_date="1980-01-01",
    end_date="2014-12-31",
    lat_min=..., lat_max=...,
    lon_min=..., lon_max=...
)
```

## Step 2: Load Observation Data
```
load_observation_data(
    variable="tas",
    source="CRU",  # or ERA5, GPCP for precipitation
    start_date="1980-01-01",
    end_date="2014-12-31",
    lat_min=..., lat_max=...,
    lon_min=..., lon_max=...
)
```

## Step 3: Regrid to Common Grid
```
regrid(
    dataset_id="ds_model",
    target_dataset_id="ds_obs",
    method="bilinear"
)
```

## Step 4: Calculate Metrics
```
# Bias
calculate_bias(
    model_dataset_id="ds_model_regridded",
    reference_dataset_id="ds_obs",
    metric="mean"
)

# Correlation
calculate_correlation(
    dataset1_id="ds_model_regridded",
    dataset2_id="ds_obs",
    correlation_type="temporal"
)

# RMSE
calculate_rmse(
    model_dataset_id="ds_model_regridded",
    reference_dataset_id="ds_obs"
)
```

## Step 5: Create Taylor Diagram
```
generate_taylor_diagram(
    model_dataset_ids=["ds_model1", "ds_model2", ...],
    reference_dataset_id="ds_obs",
    labels=["Model 1", "Model 2", ...]
)
```
"""


@mcp.prompt()
def precipitation_extremes() -> str:
    """
    Prompt template for analyzing precipitation extremes.
    """
    return """
# Precipitation Extremes Analysis Workflow

To analyze changes in precipitation extremes:

## Step 1: Load Daily Precipitation Data
```
load_climate_data(
    variable="pr",
    model="GFDL-ESM4",
    scenario="ssp370",
    start_date="2015-01-01",
    end_date="2100-12-31",
    lat_min=..., lat_max=...,
    lon_min=..., lon_max=...
)
```

## Step 2: Convert Units
```
convert_units(dataset_id="ds_xxx", target_unit="mm/day")
```

## Step 3: Calculate Extreme Indices
```
# Maximum 1-day precipitation
calculate_etccdi_index(dataset_id="ds_xxx", index="Rx1day", freq="YS")

# Maximum 5-day precipitation
calculate_etccdi_index(dataset_id="ds_xxx", index="Rx5day", freq="YS")

# Heavy precipitation days
calculate_etccdi_index(dataset_id="ds_xxx", index="R20mm", freq="YS")

# Consecutive dry days
calculate_etccdi_index(dataset_id="ds_xxx", index="CDD", freq="YS")
```

## Step 4: Analyze Trends
```
calculate_trend(dataset_id="ds_rx1day")
```

## Step 5: Visualize
```
generate_timeseries_plot(
    dataset_ids=["ds_rx1day"],
    title="Annual Maximum 1-day Precipitation",
    show_trend=True
)
```
"""


@mcp.prompt()
def drought_analysis() -> str:
    """
    Prompt template for drought analysis using SPI/SPEI.
    """
    return """
# Drought Analysis Workflow

To analyze drought conditions and trends:

## Step 1: Load Precipitation Data
```
load_climate_data(
    variable="pr",
    model="CESM2",
    scenario="ssp245",
    start_date="2015-01-01",
    end_date="2100-12-31",
    lat_min=..., lat_max=...,
    lon_min=..., lon_max=...
)
```

## Step 2: Load Temperature Data (for SPEI)
```
load_climate_data(
    variable="tas",
    model="CESM2",
    scenario="ssp245",
    start_date="2015-01-01",
    end_date="2100-12-31",
    lat_min=..., lat_max=...,
    lon_min=..., lon_max=...
)
```

## Step 3: Calculate Drought Index
```
# SPI (precipitation only)
calculate_drought_index(
    precipitation_dataset_id="ds_pr",
    index="SPI",
    scale=3  # 3-month SPI
)

# SPEI (includes temperature/evaporation effects)
calculate_drought_index(
    precipitation_dataset_id="ds_pr",
    index="SPEI",
    scale=6,
    temperature_dataset_id="ds_tas"
)
```

## Step 4: Analyze Drought Trends
```
calculate_regional_mean(dataset_id="ds_spi3", area_weighted=True)
calculate_trend(dataset_id="ds_regional_mean")
```

## Interpretation
- SPI/SPEI < -1: Moderate drought
- SPI/SPEI < -1.5: Severe drought
- SPI/SPEI < -2: Extreme drought
"""


@mcp.prompt()
def extreme_event_statistics() -> str:
    """
    Prompt template for ARSET Part 2 - Extreme Event Statistics workflow.

    Computes ETCCDI indices from NEX-GDDP-CMIP6 data, masks to country
    boundaries, and analyzes trends over the 21st century.
    """
    return """
# Extreme Event Statistics Workflow (ARSET Part 2)

Analyze extreme climate events using ETCCDI indices from NEX-GDDP-CMIP6 data,
masked to a country/region of interest.

## Step 1: Load Daily Climate Data
```
# Load daily maximum temperature for a region covering your country
load_climate_data(
    variable="tasmax",
    model="ACCESS-CM2",
    scenario="ssp585",
    start_date="2015-01-01",
    end_date="2100-12-31",
    lat_min=5, lat_max=21,    # e.g. Thailand
    lon_min=97, lon_max=106
)

# Also load daily minimum temperature and precipitation if needed
load_climate_data(variable="tasmin", ...)
load_climate_data(variable="pr", ...)
```

## Step 2: Calculate ETCCDI Indices (Batch)
```
# Temperature extremes
calculate_batch_etccdi(
    dataset_id="ds_tasmax",
    indices=["TXx", "TX90p", "SU", "WSDI"],
    freq="YS"
)

# Precipitation extremes
calculate_batch_etccdi(
    dataset_id="ds_pr",
    indices=["Rx1day", "Rx5day", "R10mm", "CDD"],
    freq="YS"
)
```

## Step 3: Mask to Country
```
# Mask each index to country boundaries
mask_by_country(dataset_id="ds_txx", country_name="Thailand")
mask_by_country(dataset_id="ds_rx1day", country_name="Thailand")
# Use list_countries() to see available country names
```

## Step 4: Compute Multi-Year Mean Climatology
```
# Calculate the mean over a reference period
calculate_climatology(dataset_id="ds_txx_masked", period="monthly")

# Or compute statistics over the full period
calculate_statistics(dataset_id="ds_txx_masked")
```

## Step 5: Calculate Area-Weighted Regional Annual Time Series
```
calculate_regional_mean(dataset_id="ds_txx_masked", area_weighted=True)
```

## Step 6: Fit Linear Trend
```
# Trend is automatically detected as annual data and gives correct per-decade slope
calculate_trend(dataset_id="ds_txx_regional_mean")
```

## Step 7: Visualize
```
# Map of mean extreme index over the country
generate_country_map(
    dataset_id="ds_txx_masked",
    country_name="Thailand",
    title="Mean Annual TXx - Thailand (SSP5-8.5)"
)

# Time series with trend
generate_timeseries_plot(
    dataset_ids=["ds_txx_regional_mean"],
    title="Annual Maximum Temperature (TXx) - Thailand",
    show_trend=True
)
```

## Interpretation
- Compare historical vs future periods to quantify changes
- Compare SSP2-4.5 vs SSP5-8.5 for scenario uncertainty
- Compare multiple models for model uncertainty
- Look for acceleration of extremes in later periods (2050-2100)
- Note: ETCCDI indices computed at annual frequency are ideal for trend analysis
"""


@mcp.prompt()
def multi_model_comparison() -> str:
    """
    Prompt template for comparing multiple climate models.
    """
    return """
# Multi-Model Comparison Workflow

To compare projections across multiple climate models:

## Step 1: Load Data from Multiple Models
```
# Load from several models
models = ["ACCESS-CM2", "CESM2", "GFDL-ESM4", "MIROC6", "MPI-ESM1-2-HR"]

for model in models:
    load_climate_data(
        variable="tas",
        model=model,
        scenario="ssp585",
        start_date="2050-01-01",
        end_date="2099-12-31",
        lat_min=..., lat_max=...,
        lon_min=..., lon_max=...
    )
```

## Step 2: Calculate Regional Means
```
for dataset_id in loaded_datasets:
    calculate_regional_mean(dataset_id=dataset_id, area_weighted=True)
```

## Step 3: Calculate Statistics for Each Model
```
for dataset_id in regional_means:
    calculate_statistics(dataset_id=dataset_id)
    calculate_trend(dataset_id=dataset_id)
```

## Step 4: Compare Visually
```
generate_timeseries_plot(
    dataset_ids=[list of regional mean dataset IDs],
    labels=models,
    title="Temperature Projections - Multi-Model Comparison",
    show_trend=True
)
```

## Step 5: Calculate Model Agreement
Look at the spread across models to understand projection uncertainty.
Model agreement is highest when all models show similar trends and magnitudes.
"""
