"""
Dataset Resources

MCP resources for exposing dataset catalogs and metadata.
"""

from __future__ import annotations

from rcmes_mcp.server import mcp
from rcmes_mcp.utils.cloud import (
    CMIP6_MODELS,
    CMIP6_SCENARIOS,
    CMIP6_VARIABLES,
)


@mcp.resource("datasets://catalog")
def get_datasets_catalog() -> str:
    """
    Get catalog of all available climate datasets.

    Returns information about NEX-GDDP-CMIP6 and other supported datasets.
    """
    return """
# Available Climate Datasets

## NEX-GDDP-CMIP6
NASA Earth Exchange Global Daily Downscaled Projections (CMIP6)

- **Size**: 38 TB
- **Resolution**: 0.25° x 0.25° (~25 km), Daily
- **Coverage**: Global (180°W-180°E, 60°S-90°N)
- **Time Period**:
  - Historical: 1950-2014
  - Projections: 2015-2100
- **Scenarios**: SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5
- **Models**: 35 CMIP6 climate models
- **Variables**: Temperature (tas, tasmax, tasmin), Precipitation (pr), Humidity, Wind, Radiation
- **Source**: https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6

## Access Methods
- AWS S3: s3://nex-gddp-cmip6 (public, no authentication required)
- OPeNDAP: Via NCCS THREDDS server

Use `list_available_models()`, `list_available_variables()`, and `list_available_scenarios()`
to explore the dataset contents.
"""


@mcp.resource("datasets://nex-gddp-cmip6/models")
def get_cmip6_models() -> str:
    """
    Get list of available CMIP6 climate models.
    """
    models_list = "\n".join(f"- {model}" for model in CMIP6_MODELS)
    return f"""
# NEX-GDDP-CMIP6 Climate Models

The following {len(CMIP6_MODELS)} climate models are available:

{models_list}

## Usage
Use these model names with the `load_climate_data()` tool:
```
load_climate_data(
    variable="tasmax",
    model="ACCESS-CM2",
    scenario="ssp585",
    ...
)
```
"""


@mcp.resource("datasets://nex-gddp-cmip6/scenarios")
def get_cmip6_scenarios() -> str:
    """
    Get list of available emissions scenarios with descriptions.
    """
    scenarios_text = ""
    for scenario_id, description in CMIP6_SCENARIOS.items():
        scenarios_text += f"\n### {scenario_id}\n{description}\n"

    return f"""
# NEX-GDDP-CMIP6 Emissions Scenarios

{scenarios_text}

## Scenario Comparison

| Scenario | Description | 2100 Warming (°C) |
|----------|-------------|-------------------|
| SSP1-2.6 | Sustainability | 1.5 - 2.0 |
| SSP2-4.5 | Middle of Road | 2.0 - 3.0 |
| SSP3-7.0 | Regional Rivalry | 3.0 - 4.0 |
| SSP5-8.5 | Fossil-fueled | 4.0 - 5.5 |

## Time Periods
- **Historical**: 1950-2014 (use scenario="historical")
- **Future**: 2015-2100 (use SSP scenarios)
"""


@mcp.resource("datasets://nex-gddp-cmip6/variables")
def get_cmip6_variables() -> str:
    """
    Get list of available climate variables with units.
    """
    var_text = ""
    for var_name, info in CMIP6_VARIABLES.items():
        var_text += f"\n### {var_name}\n- **Name**: {info['long_name']}\n- **Units**: {info['units']}\n"

    return f"""
# NEX-GDDP-CMIP6 Climate Variables

{var_text}

## Common Use Cases

### Temperature Analysis
- Use `tasmax` for heat extremes, summer days
- Use `tasmin` for frost days, cold nights
- Use `tas` for mean temperature trends

### Precipitation Analysis
- Use `pr` for rainfall totals, drought indices
- Note: pr is in kg m-2 s-1, convert to mm/day by multiplying by 86400

### Other Variables
- `hurs`: Relative humidity for heat stress analysis
- `sfcWind`: Wind speed for renewable energy assessment
- `rsds`/`rlds`: Radiation for solar energy, evaporation
"""


@mcp.resource("datasets://regions/us-states")
def get_us_states() -> str:
    """
    Get US state bounding boxes for subsetting.
    """
    return """
# US State Bounding Boxes

Common bounding boxes for US states (approximate):

| State | lat_min | lat_max | lon_min | lon_max |
|-------|---------|---------|---------|---------|
| California | 32.5 | 42.0 | -124.5 | -114.0 |
| Texas | 25.8 | 36.5 | -106.6 | -93.5 |
| Florida | 24.5 | 31.0 | -87.6 | -80.0 |
| New York | 40.5 | 45.0 | -79.8 | -71.8 |
| Arizona | 31.3 | 37.0 | -114.8 | -109.0 |
| Nevada | 35.0 | 42.0 | -120.0 | -114.0 |
| Colorado | 37.0 | 41.0 | -109.0 | -102.0 |
| Washington | 45.5 | 49.0 | -124.8 | -116.9 |
| Oregon | 42.0 | 46.3 | -124.6 | -116.5 |

## Example Usage
```
load_climate_data(
    variable="tasmax",
    model="CESM2",
    scenario="ssp585",
    start_date="2050-01-01",
    end_date="2050-12-31",
    lat_min=32.5, lat_max=42.0,
    lon_min=-124.5, lon_max=-114.0
)
```
"""
