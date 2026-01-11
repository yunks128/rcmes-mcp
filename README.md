# RCMES-MCP

**Regional Climate Model Evaluation System as an MCP Server**

RCMES-MCP provides AI agents with tools to analyze NASA's climate datasets through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). It enables conversational interaction with massive climate data (NEX-GDDP-CMIP6: 38TB) without requiring users to download data or write Python scripts.

## Features

- **Data Access**: Direct access to NEX-GDDP-CMIP6 dataset on AWS S3
- **Climate Analysis**: Climatology, trends, anomalies, regional statistics
- **Extreme Indices**: ETCCDI climate indices (heatwaves, drought, extremes)
- **Model Evaluation**: Bias, RMSE, correlation metrics, Taylor diagrams
- **Visualization**: Maps, time series, comparison plots

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/rcmes-mcp.git
cd rcmes-mcp

# Install with pip
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Running with Gemini API (Recommended)

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key"

# Run interactive chat
rcmes-gemini

# Or as Python module
python -m rcmes_mcp.gemini_client
```

### Running as MCP Server (for Claude)

```bash
# Run with stdio transport (for Claude Desktop, Claude Code)
rcmes-mcp

# Or as a Python module
python -m rcmes_mcp.server
```

### Docker

```bash
# Build the image
docker build -t rcmes-mcp .

# Run the container
docker run -p 8000:8000 rcmes-mcp
```

## Available Tools

### Data Access
| Tool | Description |
|------|-------------|
| `list_available_models` | List CMIP6 climate models |
| `list_available_variables` | List climate variables (tas, pr, etc.) |
| `list_available_scenarios` | List emissions scenarios (SSP1-2.6 to SSP5-8.5) |
| `load_climate_data` | Load data for a region and time period |
| `get_dataset_metadata` | Get metadata for a model/scenario/variable |

### Data Processing
| Tool | Description |
|------|-------------|
| `temporal_subset` | Subset by time period |
| `spatial_subset` | Subset by geographic bounds |
| `temporal_resample` | Resample to monthly/seasonal/annual |
| `convert_units` | Convert units (K→°C, kg/m²/s→mm/day) |
| `regrid` | Regrid to different resolution |
| `calculate_anomaly` | Calculate anomalies from baseline |

### Analysis
| Tool | Description |
|------|-------------|
| `calculate_statistics` | Mean, std, min, max, percentiles |
| `calculate_climatology` | Daily/monthly/seasonal climatology |
| `calculate_trend` | Linear trend with significance testing |
| `calculate_regional_mean` | Area-weighted regional average |
| `calculate_bias` | Model bias vs. reference |
| `calculate_correlation` | Temporal/spatial correlation |
| `calculate_rmse` | Root mean square error |

### Climate Indices
| Tool | Description |
|------|-------------|
| `calculate_etccdi_index` | ETCCDI indices (TX90p, Rx5day, CDD, etc.) |
| `analyze_heatwaves` | Heatwave frequency, duration, intensity |
| `calculate_drought_index` | SPI/SPEI drought indices |
| `calculate_growing_degree_days` | Agricultural growing degree days |

### Visualization
| Tool | Description |
|------|-------------|
| `generate_map` | Spatial map visualization |
| `generate_timeseries_plot` | Time series with optional trend |
| `generate_comparison_map` | Side-by-side model comparison |
| `generate_taylor_diagram` | Model evaluation diagram |
| `generate_histogram` | Data distribution histogram |

## Example Usage

### With Claude Code or other MCP clients

```
User: What is the heatwave trend in California under the high emission scenario?

Agent: Let me analyze that for you.

1. Loading daily maximum temperature data for California from 2015-2100
   under SSP5-8.5 using the ACCESS-CM2 model...

2. Converting temperature to Celsius...

3. Analyzing heatwaves (days > 90th percentile, minimum 3 consecutive days)...

4. Calculating trend...

Results: California is projected to experience a significant increase in
heatwave days under SSP5-8.5. The trend shows an increase of approximately
8.5 heatwave days per decade (p < 0.001). By 2100, the average annual
heatwave days could increase from ~15 to ~85 days.

[Generated time series plot showing the trend]
```

## Dataset Information

### NEX-GDDP-CMIP6
- **Size**: 38 TB
- **Resolution**: 0.25° x 0.25° (~25 km), Daily
- **Coverage**: Global (180°W-180°E, 60°S-90°N)
- **Time Period**: 1950-2100
- **Scenarios**: Historical, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5
- **Models**: 35 CMIP6 climate models
- **Variables**: tas, tasmax, tasmin, pr, hurs, huss, sfcWind, rsds, rlds
- **Source**: [NASA NCCS](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6)

## Configuration

Environment variables:
- `RCMES_CACHE_DIR`: Directory for result caching (default: system temp)
- `RCMES_SESSION_TTL`: Session dataset TTL in hours (default: 2)
- `DASK_SCHEDULER`: Dask scheduler address for distributed computing

## Architecture

```
Azure AI Foundry / Claude / Other LLM
           │
           │ MCP Protocol
           ▼
    ┌─────────────────┐
    │  RCMES-MCP      │
    │  (FastMCP)      │
    │  ┌───────────┐  │
    │  │   Tools   │  │
    │  │ Resources │  │
    │  │  Prompts  │  │
    │  └───────────┘  │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
 AWS S3           RCMED
NEX-GDDP-CMIP6   Observations
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/

# Type checking
mypy src/
```

## License

Apache License 2.0

## References

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Apache Open Climate Workbench](https://climate.apache.org/)
- [NEX-GDDP-CMIP6 on AWS](https://registry.opendata.aws/nex-gddp-cmip6/)
- [xclim Climate Indices](https://xclim.readthedocs.io/)
- [ETCCDI Climate Indices](http://etccdi.pacificclimate.org/list_27_indices.shtml)

## Contact

- NASA JPL RCMES: https://rcmes.jpl.nasa.gov/
- NCCS Support: support@nccs.nasa.gov
