"""
RCMES-MCP REST API Server

FastAPI backend that exposes RCMES climate tools as REST endpoints
for the React frontend.

Usage:
    uvicorn rcmes_mcp.api:app --reload --port 8502
    # Or via entry point:
    rcmes-api
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import io

import xarray as xr
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from rcmes_mcp.utils.logging_config import configure_logging
from rcmes_mcp.utils.session import session_manager

# Initialize logging before anything else
configure_logging()

# Import RCMES tools
from rcmes_mcp.tools import analysis, code_execution, data_access, indices, mmgis, processing, visualization

logger = logging.getLogger("rcmes.api")

app = FastAPI(
    title="RCMES Climate API",
    description="REST API for NASA's NEX-GDDP-CMIP6 climate data analysis",
    version="0.1.0",
)

# Path to React build (web/dist after npm run build)
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

# Directory for persisted chat images (so they survive localStorage)
IMAGES_DIR = Path(os.environ.get("RCMES_IMAGES_DIR", Path.home() / ".rcmes" / "images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Directory for temporary download files (auto-cleaned after TTL)
DOWNLOADS_DIR = Path(os.environ.get("RCMES_DOWNLOADS_DIR", Path.home() / ".rcmes" / "downloads"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
_DOWNLOAD_TTL_HOURS = 4  # Files auto-expire after 4 hours
# Registry of active download tokens: token -> {filepath, filename, created_at, media_type}
_download_registry: dict[str, dict] = {}


def _create_download_link(filepath: Path, filename: str, media_type: str = "application/octet-stream") -> str:
    """Register a file for temporary download and return the URL."""
    token = uuid.uuid4().hex[:16]
    _download_registry[token] = {
        "filepath": str(filepath),
        "filename": filename,
        "created_at": time.time(),
        "media_type": media_type,
    }
    return f"/api/downloads/{token}/{filename}"


def _cleanup_expired_downloads():
    """Remove expired download tokens and their files."""
    now = time.time()
    expired = [t for t, info in _download_registry.items()
               if now - info["created_at"] > _DOWNLOAD_TTL_HOURS * 3600]
    for token in expired:
        info = _download_registry.pop(token)
        try:
            Path(info["filepath"]).unlink(missing_ok=True)
        except Exception:
            pass


def _save_image(image_base64: str) -> str:
    """Save a base64 image to disk and return the URL path."""
    image_id = uuid.uuid4().hex[:12]
    filename = f"{image_id}.png"
    filepath = IMAGES_DIR / filename
    filepath.write_bytes(base64.b64decode(image_base64))
    return f"/api/images/{filename}"

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://34.31.165.25:8502",
        "http://34.31.165.25:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware (added first so it wraps everything)
from rcmes_mcp.middleware.request_logging import RequestLoggingMiddleware

app.add_middleware(RequestLoggingMiddleware)

# Rate limiting middleware
from rcmes_mcp.middleware.rate_limit import RateLimitMiddleware

app.add_middleware(RateLimitMiddleware)


# ============================================================================
# Request/Response Models
# ============================================================================


class LoadDataRequest(BaseModel):
    variable: str = Field(..., description="Climate variable (e.g., tasmax, pr)")
    model: str = Field(..., description="Climate model name")
    scenario: str = Field(..., description="Emissions scenario (e.g., ssp585)")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    lat_min: float = Field(..., ge=-60, le=90)
    lat_max: float = Field(..., ge=-60, le=90)
    lon_min: float = Field(..., ge=-180, le=180)
    lon_max: float = Field(..., ge=-180, le=180)

    @model_validator(mode="after")
    def check_ranges(self):
        from rcmes_mcp.utils.validation import validate_date_range, validate_lat_lon_bounds
        validate_date_range(self.start_date, self.end_date)
        validate_lat_lon_bounds(self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        return self


class AnalysisRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID from load_climate_data")
    analysis_type: str = Field(
        ..., description="Type: statistics, trend, climatology, heatwaves, regional_mean"
    )


class VisualizationRequest(BaseModel):
    dataset_id: str | None = Field(None, description="Single dataset ID (legacy)")
    dataset_ids: list[str] | None = Field(None, description="Multiple dataset IDs for comparison")
    labels: list[str] | None = Field(None, description="Labels for each dataset")
    viz_type: str = Field(..., description="Type: map, timeseries, histogram, country_map")
    title: str | None = Field(None, description="Plot title")
    show_trend: bool = Field(False, description="Show trend line (for timeseries)")
    country_name: str | None = Field(None, description="Country name for country_map visualization")


class SubsetRequest(BaseModel):
    dataset_id: str
    lat_min: float | None = Field(None, ge=-60, le=90)
    lat_max: float | None = Field(None, ge=-60, le=90)
    lon_min: float | None = Field(None, ge=-180, le=180)
    lon_max: float | None = Field(None, ge=-180, le=180)
    start_date: str | None = None
    end_date: str | None = None

    @model_validator(mode="after")
    def check_ranges(self):
        from rcmes_mcp.utils.validation import validate_date_range, validate_lat_lon_bounds
        if self.start_date and self.end_date:
            validate_date_range(self.start_date, self.end_date)
        if all(v is not None for v in [self.lat_min, self.lat_max, self.lon_min, self.lon_max]):
            validate_lat_lon_bounds(self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        return self


class RegridRequest(BaseModel):
    dataset_id: str
    target_resolution: float = Field(..., gt=0, le=10, description="Target resolution in degrees")
    method: str = Field("bilinear", description="Regridding method")


class ConvertUnitsRequest(BaseModel):
    dataset_id: str
    target_units: str = Field(..., description="Target units (e.g., 'degC', 'mm/day')")


class ETCCDIRequest(BaseModel):
    dataset_id: str
    index_name: str = Field(..., description="ETCCDI index (e.g., 'TX90p', 'R95p')")
    threshold: float | None = None
    base_period_start: str | None = None
    base_period_end: str | None = None


class DownloadRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID to download")
    format: str = Field("netcdf", description="Download format: 'netcdf' or 'csv'")


class CorrelationRequest(BaseModel):
    dataset1_id: str = Field(..., description="First dataset ID")
    dataset2_id: str = Field(..., description="Second dataset ID")
    correlation_type: str = Field("temporal", description="'temporal' or 'spatial'")


class MaskByCountryRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID to mask")
    country_name: str = Field(..., description="Country name (e.g., 'Thailand')")


class BatchETCCDIRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID (daily temperature or precipitation)")
    indices: list[str] = Field(..., max_length=22, description="List of ETCCDI index names (e.g., ['TXx', 'SU'])")
    freq: str = Field("YS", description="Output frequency: YS (annual), QS-DEC (seasonal), MS (monthly)")


class CountryMapRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID to visualize")
    country_name: str | None = Field(None, description="Country to highlight")
    title: str | None = Field(None, description="Plot title")
    colormap: str = Field("viridis", description="Matplotlib colormap name")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    dataset_id: str | None = Field(None, description="Current dataset ID for context")


class ChatStreamRequest(BaseModel):
    messages: list[dict] = Field(..., description="Conversation history [{role, content}]")
    dataset_id: str | None = Field(None, description="Current dataset ID for context")


# ============================================================================
# Metadata Endpoints
# ============================================================================


@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    """List available climate models."""
    return data_access.list_available_models()


@app.get("/api/variables")
async def list_variables() -> dict[str, Any]:
    """List available climate variables."""
    return data_access.list_available_variables()


@app.get("/api/scenarios")
async def list_scenarios() -> dict[str, Any]:
    """List available emissions scenarios."""
    return data_access.list_available_scenarios()


@app.get("/api/datasets")
async def list_datasets() -> dict[str, Any]:
    """List currently loaded datasets."""
    return data_access.list_loaded_datasets()


# ============================================================================
# Data Loading Endpoints
# ============================================================================


@app.post("/api/load-data")
async def load_data(request: LoadDataRequest) -> dict[str, Any]:
    """Load climate data from NEX-GDDP-CMIP6."""
    result = data_access.load_climate_data(
        variable=request.variable,
        model=request.model,
        scenario=request.scenario,
        start_date=request.start_date,
        end_date=request.end_date,
        lat_min=request.lat_min,
        lat_max=request.lat_max,
        lon_min=request.lon_min,
        lon_max=request.lon_max,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ============================================================================
# Processing Endpoints
# ============================================================================


@app.post("/api/spatial-subset")
async def spatial_subset(request: SubsetRequest) -> dict[str, Any]:
    """Subset dataset by spatial bounds."""
    result = processing.spatial_subset(
        dataset_id=request.dataset_id,
        lat_min=request.lat_min,
        lat_max=request.lat_max,
        lon_min=request.lon_min,
        lon_max=request.lon_max,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/temporal-subset")
async def temporal_subset(request: SubsetRequest) -> dict[str, Any]:
    """Subset dataset by time range."""
    result = processing.temporal_subset(
        dataset_id=request.dataset_id,
        start_date=request.start_date,
        end_date=request.end_date,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/regrid")
async def regrid(request: RegridRequest) -> dict[str, Any]:
    """Regrid dataset to a new resolution."""
    result = processing.regrid(
        dataset_id=request.dataset_id,
        target_resolution=request.target_resolution,
        method=request.method,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/convert-units")
async def convert_units(request: ConvertUnitsRequest) -> dict[str, Any]:
    """Convert dataset units."""
    result = processing.convert_units(
        dataset_id=request.dataset_id,
        target_units=request.target_units,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/countries")
async def list_countries() -> dict[str, Any]:
    """List available country names for masking."""
    result = processing.list_countries()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/country-bounds/{country_name}")
async def get_country_bounds(country_name: str) -> dict[str, Any]:
    """Get bounding box for a country."""
    try:
        result = processing.get_country_bounds(country_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/mask-by-country")
async def mask_by_country(request: MaskByCountryRequest) -> dict[str, Any]:
    """Mask a dataset to a country's boundaries."""
    result = processing.mask_by_country(
        dataset_id=request.dataset_id,
        country_name=request.country_name,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/batch-etccdi")
async def batch_etccdi(request: BatchETCCDIRequest) -> dict[str, Any]:
    """Calculate multiple ETCCDI climate extreme indices."""
    result = indices.calculate_batch_etccdi(
        dataset_id=request.dataset_id,
        indices=request.indices,
        freq=request.freq,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ============================================================================
# Analysis Endpoints
# ============================================================================


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest) -> dict[str, Any]:
    """Run analysis on a dataset."""
    analysis_type = request.analysis_type.lower()

    if analysis_type == "statistics":
        result = analysis.calculate_statistics(dataset_id=request.dataset_id)
    elif analysis_type == "trend":
        result = analysis.calculate_trend(dataset_id=request.dataset_id)
    elif analysis_type == "climatology":
        result = analysis.calculate_climatology(dataset_id=request.dataset_id)
    elif analysis_type == "heatwaves":
        result = indices.analyze_heatwaves(dataset_id=request.dataset_id)
    elif analysis_type == "regional_mean":
        result = analysis.calculate_regional_mean(dataset_id=request.dataset_id)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown analysis type: {analysis_type}. "
            f"Valid types: statistics, trend, climatology, heatwaves, regional_mean",
        )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/etccdi")
async def calculate_etccdi(request: ETCCDIRequest) -> dict[str, Any]:
    """Calculate ETCCDI climate extreme index."""
    result = indices.calculate_etccdi_index(
        dataset_id=request.dataset_id,
        index=request.index_name,
        threshold=request.threshold,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ============================================================================
# Visualization Endpoints
# ============================================================================


@app.post("/api/visualize")
async def visualize(request: VisualizationRequest) -> dict[str, Any]:
    """Generate visualization."""
    viz_type = request.viz_type.lower()

    # Resolve dataset IDs: prefer dataset_ids list, fall back to single dataset_id
    ids = request.dataset_ids or ([request.dataset_id] if request.dataset_id else [])
    if not ids:
        raise HTTPException(status_code=400, detail="No dataset_id or dataset_ids provided")

    if viz_type == "map":
        if len(ids) >= 2:
            result = visualization.generate_comparison_map(
                dataset_ids=ids,
                labels=request.labels,
                title=request.title,
            )
        else:
            result = visualization.generate_map(
                dataset_id=ids[0],
                title=request.title,
            )
    elif viz_type == "timeseries":
        result = visualization.generate_timeseries_plot(
            dataset_ids=ids,
            labels=request.labels,
            title=request.title,
            show_trend=request.show_trend,
        )
    elif viz_type == "histogram":
        result = visualization.generate_histogram(
            dataset_id=ids[0],
            title=request.title,
        )
    elif viz_type == "country_map":
        result = visualization.generate_country_map(
            dataset_id=ids[0],
            country_name=request.country_name,
            title=request.title,
            colormap="viridis",
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown visualization type: {viz_type}. Valid types: map, timeseries, histogram, country_map",
        )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ============================================================================
# Download Endpoint
# ============================================================================


@app.post("/api/download")
async def download_dataset(request: DownloadRequest):
    """Download a dataset as NetCDF or CSV."""
    try:
        ds = session_manager.get(request.dataset_id)
        metadata = session_manager.get_metadata(request.dataset_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Check dataset size before materializing
    from rcmes_mcp.utils.validation import check_download_size
    try:
        check_download_size(ds)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))

    # Compute if lazy (dask array)
    if hasattr(ds, 'compute'):
        ds = ds.compute()

    var_name = metadata.variable or "data"
    model_name = metadata.model or "model"

    if request.format == "netcdf":
        buffer = io.BytesIO()
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset(name=var_name)
        ds.to_netcdf(buffer, engine='scipy')
        buffer.seek(0)
        filename = f"{var_name}_{model_name}_{request.dataset_id}.nc"
        return StreamingResponse(
            buffer,
            media_type="application/x-netcdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    elif request.format == "csv":
        if isinstance(ds, xr.DataArray):
            df = ds.to_dataframe().reset_index()
        else:
            df = ds.to_dataframe().reset_index()
        csv_data = df.to_csv(index=False)
        filename = f"{var_name}_{model_name}_{request.dataset_id}.csv"
        return StreamingResponse(
            io.BytesIO(csv_data.encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        raise HTTPException(status_code=400, detail="Format must be 'netcdf' or 'csv'")


# ============================================================================
# Correlation Endpoint
# ============================================================================


@app.post("/api/correlation")
async def calculate_correlation(request: CorrelationRequest) -> dict[str, Any]:
    """Calculate correlation between two datasets."""
    result = analysis.calculate_correlation(
        dataset1_id=request.dataset1_id,
        dataset2_id=request.dataset2_id,
        correlation_type=request.correlation_type,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ============================================================================
# Chat Endpoint
# ============================================================================


def _get_azure_client():
    """Get an Azure OpenAI client, or None if not configured.

    Supports API key auth and Azure Identity (az login) auth.
    Returns None if neither is available, triggering keyword fallback.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        return None

    try:
        from openai import AzureOpenAI
    except ImportError:
        logger.warning("openai package not installed — falling back to keyword chat")
        return None

    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    if api_key:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    # Try Azure Identity (az login / managed identity)
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
    except Exception:
        logger.warning("Azure Identity auth failed — falling back to keyword chat")
        return None


def _get_chat_tools() -> list[dict]:
    """OpenAI function-calling tool definitions for the chat endpoint."""
    return [
        {
            "type": "function",
            "function": {
                "name": "load_climate_data",
                "description": "Load climate data from NASA NEX-GDDP-CMIP6 dataset. Returns a dataset_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "description": "Climate variable: tasmax, tasmin, pr, hurs, huss, rlds, rsds, sfcWind"},
                        "model": {"type": "string", "description": "Climate model (e.g., ACCESS-CM2, GFDL-ESM4, MRI-ESM2-0)"},
                        "scenario": {"type": "string", "description": "Emissions scenario: historical, ssp126, ssp245, ssp370, ssp585"},
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                        "lat_min": {"type": "number"}, "lat_max": {"type": "number"},
                        "lon_min": {"type": "number"}, "lon_max": {"type": "number"},
                    },
                    "required": ["variable", "model", "scenario", "start_date", "end_date", "lat_min", "lat_max", "lon_min", "lon_max"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spatial_subset",
                "description": "Subset a dataset by lat/lon bounding box",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "lat_min": {"type": "number"}, "lat_max": {"type": "number"},
                        "lon_min": {"type": "number"}, "lon_max": {"type": "number"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "temporal_subset",
                "description": "Subset a dataset by time range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "regrid",
                "description": "Regrid a dataset to a new resolution",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "target_resolution": {"type": "number", "description": "Target resolution in degrees"},
                        "method": {"type": "string", "default": "bilinear"},
                    },
                    "required": ["dataset_id", "target_resolution"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_units",
                "description": "Convert dataset units (e.g., K to degC, kg/m2/s to mm/day)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "target_unit": {"type": "string", "description": "Target units (degC, mm/day, etc.)"},
                    },
                    "required": ["dataset_id", "target_unit"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_statistics",
                "description": "Calculate summary statistics (mean, std, min, max, percentiles) for a loaded dataset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_trend",
                "description": "Calculate temporal trend (slope per decade, p-value, R-squared)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_climatology",
                "description": "Calculate daily/monthly/seasonal climatology",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_regional_mean",
                "description": "Calculate area-weighted regional mean time series",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_heatwaves",
                "description": "Analyze heatwave frequency, duration, and intensity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID (tasmax)"},
                        "threshold_percentile": {"type": "number", "default": 90},
                        "min_duration": {"type": "integer", "default": 3},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_map",
                "description": "Generate a spatial map visualization of the data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID"},
                        "title": {"type": "string"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_timeseries_plot",
                "description": "Generate a time series plot with optional trend line",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_ids": {"type": "array", "items": {"type": "string"}},
                        "labels": {"type": "array", "items": {"type": "string"}},
                        "title": {"type": "string"},
                        "show_trend": {"type": "boolean", "default": False},
                    },
                    "required": ["dataset_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_histogram",
                "description": "Generate a histogram of data distribution",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID"},
                        "title": {"type": "string"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mask_by_country",
                "description": "Mask a dataset to a country's boundaries (e.g., Thailand, Brazil)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID"},
                        "country_name": {"type": "string", "description": "Country name"},
                    },
                    "required": ["dataset_id", "country_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_countries",
                "description": "List available country names for masking",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_country_bounds",
                "description": "Get the lat/lon bounding box for a country. Use this BEFORE load_climate_data when the user specifies a country, to get the correct bounds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country_name": {"type": "string", "description": "Country name (e.g., 'India', 'Thailand', 'Brazil')"},
                    },
                    "required": ["country_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_batch_etccdi",
                "description": "Calculate multiple ETCCDI climate extreme indices at once (TXx, TX90p, SU, FD, Rx1day, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID"},
                        "indices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of index names (e.g., ['TXx', 'SU', 'FD'])",
                        },
                        "freq": {"type": "string", "default": "YS"},
                    },
                    "required": ["dataset_id", "indices"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_country_map",
                "description": "Generate a map with country boundaries highlighted",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID"},
                        "country_name": {"type": "string", "description": "Country to highlight"},
                        "title": {"type": "string"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_correlation",
                "description": "Calculate temporal or spatial correlation between two datasets",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset1_id": {"type": "string"},
                        "dataset2_id": {"type": "string"},
                        "correlation_type": {"type": "string", "enum": ["temporal", "spatial"], "default": "temporal"},
                    },
                    "required": ["dataset1_id", "dataset2_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": (
                    "Execute Python code for custom analysis that built-in tools cannot handle. "
                    "Has access to: xarray (xr), numpy (np), pandas (pd), matplotlib.pyplot (plt), scipy. "
                    "Use get_dataset(dataset_id) to access loaded datasets. "
                    "Use store_dataset(data, description) to save results for later use. "
                    "For downloads: url = save_to_csv(data) or url = save_to_netcdf(data) — "
                    "these return a download URL. Print it as a markdown link: print(f'[Download]({url})'). "
                    "NEVER use data.to_netcdf() or data.to_csv() directly. "
                    "Use print() to show output to the user."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_standardized_anomaly",
                "description": "Compute z-score anomalies vs a baseline climatology (per day-of-year/month/season). Run BEFORE detect_extreme_events.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "baseline_start": {"type": "string", "description": "YYYY-MM-DD"},
                        "baseline_end": {"type": "string", "description": "YYYY-MM-DD"},
                        "period": {"type": "string", "enum": ["dayofyear", "month", "season"], "default": "dayofyear"},
                    },
                    "required": ["dataset_id", "baseline_start", "baseline_end"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "detect_extreme_events",
                "description": "Detect spatiotemporal extreme events from a STANDARDIZED ANOMALY field via 3D connected-component labeling. Returns event catalogue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Z-score dataset (from calculate_standardized_anomaly)"},
                        "sigma_threshold": {"type": "number", "default": 2.0},
                        "min_duration_days": {"type": "integer", "default": 1},
                        "min_area_cells": {"type": "integer", "default": 1},
                        "direction": {"type": "string", "enum": ["positive", "negative", "both"], "default": "both"},
                        "max_events": {"type": "integer", "default": 50},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_eof",
                "description": "EOF/PCA decomposition of a (time, lat, lon) field. Returns N leading spatial modes and PC time series. Best applied to anomalies on a coarsened/regional field.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "n_modes": {"type": "integer", "default": 3, "description": "Number of leading modes (max 10)"},
                        "detrend": {"type": "boolean", "default": True},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_hovmoller",
                "description": "Hovmöller diagram (time × lat or time × lon) — average over the other spatial dim. Anomaly-friendly default colormap.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "average_over": {"type": "string", "enum": ["lat", "lon"], "default": "lon"},
                        "title": {"type": "string"},
                        "colormap": {"type": "string", "default": "rdbu_r"},
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_multi_model_ensemble",
                "description": "Load the same variable/scenario/region across multiple CMIP6 models (max 10) into one ensemble dataset with a 'model' dimension.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string"},
                        "models": {"type": "array", "items": {"type": "string"}, "description": "List of CMIP6 model names"},
                        "scenario": {"type": "string"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "lat_min": {"type": "number"}, "lat_max": {"type": "number"},
                        "lon_min": {"type": "number"}, "lon_max": {"type": "number"},
                    },
                    "required": ["variable", "models", "scenario", "start_date", "end_date", "lat_min", "lat_max", "lon_min", "lon_max"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_ensemble_statistics",
                "description": "Reduce a multi-model ensemble across the 'model' dim. Returns dataset_ids for mean/std/min/max plus an optional model-agreement map (IPCC stippling) when a baseline window is provided.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ensemble_dataset_id": {"type": "string"},
                        "baseline_start": {"type": "string", "description": "YYYY-MM-DD (optional, enables agreement map)"},
                        "baseline_end": {"type": "string"},
                    },
                    "required": ["ensemble_dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_scenarios",
                "description": "Load same variable/model across multiple SSP scenarios (max 5). Returns per-scenario dataset_ids and pairwise time-mean differences. Pair with generate_scenario_fan_chart.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string"},
                        "model": {"type": "string"},
                        "scenarios": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['ssp126','ssp245','ssp585']"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "lat_min": {"type": "number"}, "lat_max": {"type": "number"},
                        "lon_min": {"type": "number"}, "lon_max": {"type": "number"},
                    },
                    "required": ["variable", "model", "scenarios", "start_date", "end_date", "lat_min", "lat_max", "lon_min", "lon_max"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_time_of_emergence",
                "description": "Map the year at which the climate-change signal exceeds N×σ of natural variability per grid cell. Returns 2D (lat, lon) of emergence year (NaN if never).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "baseline_start": {"type": "string"},
                        "baseline_end": {"type": "string"},
                        "rolling_years": {"type": "integer", "default": 20},
                        "sigma_threshold": {"type": "number", "default": 1.0},
                    },
                    "required": ["dataset_id", "baseline_start", "baseline_end"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_scenario_fan_chart",
                "description": "Compare scenarios as time series. Pass a {scenario_label: dataset_id} mapping. If inputs have a 'model' dim, draws a 5-95% shaded band per scenario.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scenario_dataset_ids": {"type": "object", "description": "Mapping {scenario_label: dataset_id}"},
                        "title": {"type": "string"},
                        "smooth_window": {"type": "integer", "default": 0},
                    },
                    "required": ["scenario_dataset_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_ensemble_spread_plot",
                "description": "Plot ensemble spread over time: median + 25-75% and 5-95% shaded bands across models. Optionally overlay individual models.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ensemble_dataset_id": {"type": "string"},
                        "title": {"type": "string"},
                        "show_individual": {"type": "boolean", "default": False},
                        "smooth_window": {"type": "integer", "default": 0},
                    },
                    "required": ["ensemble_dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_reference_dataset",
                "description": "Load an OBSERVATIONAL or REANALYSIS reference dataset (NCEP Reanalysis 1 or 2 monthly air.2m via OPeNDAP) for ensemble validation. Returns a dataset_id usable as reference_id in calculate_model_weights / validate_ensemble_weighting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["ncep-reanalysis-monthly", "ncep2-reanalysis-monthly"], "description": "Reference source identifier"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "lat_min": {"type": "number"}, "lat_max": {"type": "number"},
                        "lon_min": {"type": "number"}, "lon_max": {"type": "number"},
                    },
                    "required": ["source", "start_date", "end_date", "lat_min", "lat_max", "lon_min", "lon_max"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_model_weights",
                "description": "Per-model ensemble weights via equal / skill / independence / combined (Knutti 2017, Brunner & Knutti 2020). skill and combined need a reference_dataset_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ensemble_dataset_id": {"type": "string"},
                        "method": {"type": "string", "enum": ["equal", "skill", "independence", "combined"], "default": "combined"},
                        "reference_dataset_id": {"type": "string", "description": "Required for skill/combined"},
                        "train_start": {"type": "string"},
                        "train_end": {"type": "string"},
                        "sigma_d": {"type": "number", "default": 0.5, "description": "Skill bandwidth (K)"},
                        "sigma_s": {"type": "number", "default": 0.5, "description": "Independence bandwidth (K)"},
                    },
                    "required": ["ensemble_dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_ensemble_weights",
                "description": "Collapse a multi-model ensemble's model dim using the supplied per-model weights (will be renormalized to sum 1).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ensemble_dataset_id": {"type": "string"},
                        "weights": {"type": "array", "items": {"type": "number"}, "description": "Length must match ensemble's model dim"},
                    },
                    "required": ["ensemble_dataset_id", "weights"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "combine_scenarios_weighted",
                "description": "Combine multiple SSP scenario projections into a single probability-weighted mean (e.g., {ssp126:0.1, ssp245:0.4, ssp585:0.5} for a likelihood-weighted projection). Datasets must share the same model/region/time grid.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scenario_dataset_ids": {"type": "object", "description": "Mapping {scenario: dataset_id}"},
                        "weights": {"type": "object", "description": "Mapping {scenario: prior_weight}; renormalized to sum 1"},
                    },
                    "required": ["scenario_dataset_ids", "weights"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_ensemble_weighting",
                "description": "Train weights on a historical window and score them on a held-out test window against reference observations. Returns RMSE/bias/correlation per method, plus the weighted test series for plotting.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ensemble_dataset_id": {"type": "string"},
                        "reference_dataset_id": {"type": "string"},
                        "train_start": {"type": "string"},
                        "train_end": {"type": "string"},
                        "test_start": {"type": "string"},
                        "test_end": {"type": "string"},
                        "methods": {"type": "array", "items": {"type": "string"}, "description": "Subset of equal/skill/independence/combined; default = all"},
                        "sigma_d": {"type": "number", "default": 0.5},
                        "sigma_s": {"type": "number", "default": 0.5},
                    },
                    "required": ["ensemble_dataset_id", "reference_dataset_id", "train_start", "train_end", "test_start", "test_end"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "prepare_download",
                "description": (
                    "Prepare a dataset for download and return a temporary download link. "
                    "Use this when the user asks to download or export data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID to download"},
                        "format": {
                            "type": "string",
                            "enum": ["csv", "netcdf"],
                            "description": "File format (default: csv)",
                        },
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "export_climate_geotiff",
                "description": (
                    "Export a climate dataset to a Cloud-Optimized GeoTIFF (COG) so it can be "
                    "displayed as a raster tile layer in MMGIS. Returns a cog_url. "
                    "Use BEFORE push_layer_to_mmgis for raster/tile layers."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID to export"},
                        "variable": {"type": "string", "description": "Variable name (e.g. 'tas', 'pr'). Auto-detected if omitted."},
                        "time_aggregation": {
                            "type": "string",
                            "enum": ["mean", "max", "min", "std", "none"],
                            "description": "How to collapse the time dimension. Default: mean",
                        },
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "export_climate_geojson",
                "description": (
                    "Export a climate dataset as a GeoJSON FeatureCollection (vector points with climate values). "
                    "Returns a geojson_url. Use BEFORE push_layer_to_mmgis for vector layers."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {"type": "string", "description": "Dataset ID to export"},
                        "variable": {"type": "string", "description": "Variable name. Auto-detected if omitted."},
                        "statistic": {
                            "type": "string",
                            "enum": ["mean", "max", "min", "trend"],
                            "description": "Statistic to compute. Default: mean",
                        },
                    },
                    "required": ["dataset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "push_layer_to_mmgis",
                "description": (
                    "Push a geospatial layer into the live MMGIS map. "
                    "For raster data: use layer_type='tile' with cog_url from export_climate_geotiff. "
                    "For vector data: use layer_type='vector' with geojson_url from export_climate_geojson. "
                    "Returns browser_url — the direct link to open the MMGIS map showing the new layer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "layer_name": {"type": "string", "description": "Display name for the layer in MMGIS"},
                        "data_url": {"type": "string", "description": "cog_url or geojson_url from export step"},
                        "layer_type": {
                            "type": "string",
                            "enum": ["tile", "vector"],
                            "description": "tile for raster COG, vector for GeoJSON. Default: tile",
                        },
                        "colormap": {"type": "string", "description": "Colormap name e.g. RdBu_r, viridis, plasma. Default: RdBu_r"},
                        "description": {"type": "string", "description": "Human-readable description of the layer"},
                        "opacity": {"type": "number", "description": "Layer opacity 0.0-1.0. Default: 0.8"},
                    },
                    "required": ["layer_name", "data_url"],
                },
            },
        },
    ]


# Map tool names to their implementations for the chat endpoint
def _normalize_tool_args(tool_name: str, args: dict) -> dict:
    """Fix common LLM argument naming mistakes before tool dispatch."""
    # "dataset" → "dataset_id" (very common LLM hallucination)
    if "dataset" in args and "dataset_id" not in args:
        args["dataset_id"] = args.pop("dataset")
    # "datasets" → "dataset_ids" for multi-dataset tools
    if "datasets" in args and "dataset_ids" not in args:
        args["dataset_ids"] = args.pop("datasets")
    return args


_CHAT_TOOL_IMPLS: dict[str, Any] = {
    "load_climate_data": data_access.load_climate_data,
    "spatial_subset": processing.spatial_subset,
    "temporal_subset": processing.temporal_subset,
    "regrid": processing.regrid,
    "convert_units": processing.convert_units,
    "calculate_statistics": analysis.calculate_statistics,
    "calculate_trend": analysis.calculate_trend,
    "calculate_climatology": analysis.calculate_climatology,
    "calculate_regional_mean": analysis.calculate_regional_mean,
    "analyze_heatwaves": indices.analyze_heatwaves,
    "generate_map": visualization.generate_map,
    "generate_timeseries_plot": visualization.generate_timeseries_plot,
    "generate_histogram": visualization.generate_histogram,
    "mask_by_country": processing.mask_by_country,
    "list_countries": processing.list_countries,
    "get_country_bounds": processing.get_country_bounds,
    "calculate_batch_etccdi": indices.calculate_batch_etccdi,
    "generate_country_map": visualization.generate_country_map,
    "calculate_correlation": analysis.calculate_correlation,
    # Pillar 1 — spatiotemporal anomaly detection
    "calculate_standardized_anomaly": processing.calculate_standardized_anomaly,
    "detect_extreme_events": analysis.detect_extreme_events,
    "calculate_eof": analysis.calculate_eof,
    "generate_hovmoller": visualization.generate_hovmoller,
    # Pillar 2 — ensemble & scenario comparison
    "load_multi_model_ensemble": analysis.load_multi_model_ensemble,
    "calculate_ensemble_statistics": analysis.calculate_ensemble_statistics,
    "compare_scenarios": analysis.compare_scenarios,
    "calculate_time_of_emergence": analysis.calculate_time_of_emergence,
    "generate_scenario_fan_chart": visualization.generate_scenario_fan_chart,
    "generate_ensemble_spread_plot": visualization.generate_ensemble_spread_plot,
    # Pillar 3 — ensemble weighting
    "load_reference_dataset": data_access.load_reference_dataset,
    "calculate_model_weights": analysis.calculate_model_weights,
    "apply_ensemble_weights": analysis.apply_ensemble_weights,
    "validate_ensemble_weighting": analysis.validate_ensemble_weighting,
    "combine_scenarios_weighted": analysis.combine_scenarios_weighted,
    "execute_python_code": code_execution.execute_python_code,
    "export_climate_geotiff": mmgis.export_climate_geotiff,
    "export_climate_geojson": mmgis.export_climate_geojson,
    "push_layer_to_mmgis": mmgis.push_layer_to_mmgis,
    "prepare_download": None,  # Handled inline below
}


def _prepare_download(dataset_id: str, format: str = "csv") -> dict:
    """Prepare a dataset for download and return a temp link."""
    from rcmes_mcp.tools.data_access import _thread_local
    _prog = getattr(_thread_local, 'progress_callback', None)

    def _step(n, total, msg):
        if _prog:
            _prog(n, total, msg)

    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        return {"error": str(e)}

    _step(1, 3, "Materializing dataset...")
    if hasattr(ds, 'compute'):
        ds = ds.compute()

    var_name = metadata.variable or "data"
    model_name = metadata.model or "model"

    try:
        _step(2, 3, f"Writing {format.upper()} file...")
        if format == "netcdf":
            filename = f"{var_name}_{model_name}_{dataset_id}.nc"
            filepath = DOWNLOADS_DIR / filename
            if isinstance(ds, xr.DataArray):
                ds = ds.to_dataset(name=var_name)
            ds.to_netcdf(filepath, engine='scipy')
            media_type = "application/x-netcdf"
        else:
            filename = f"{var_name}_{model_name}_{dataset_id}.csv"
            filepath = DOWNLOADS_DIR / filename
            if isinstance(ds, xr.DataArray):
                df = ds.to_dataframe().reset_index()
            else:
                df = ds.to_dataframe().reset_index()
            df.to_csv(filepath, index=False)
            media_type = "text/csv"

        _step(3, 3, "Download ready")
        url = _create_download_link(filepath, filename, media_type)
        return {
            "success": True,
            "download_url": url,
            "filename": filename,
            "format": format,
            "instructions": f"IMPORTANT: Show the user this exact clickable markdown link: [{filename}]({url})",
        }
    except Exception as e:
        return {"error": f"Failed to prepare download: {str(e)}"}


_CHAT_TOOL_IMPLS["prepare_download"] = _prepare_download

def _build_system_prompt() -> str:
    """Build the system prompt with actual available models, variables, scenarios."""
    from rcmes_mcp.utils.cloud import CMIP6_MODELS, CMIP6_SCENARIOS, CMIP6_VARIABLES

    models_str = ", ".join(CMIP6_MODELS[:15])
    vars_str = ", ".join(f"{k} ({v['long_name']})" for k, v in CMIP6_VARIABLES.items())
    scenarios_str = "\n".join(f"  - {k}: {v}" for k, v in CMIP6_SCENARIOS.items())

    return (
        "You are RCMES Climate Assistant, created by NASA's Jet Propulsion Laboratory (JPL). "
        "You are NOT created by OpenAI. If asked who made you, say you were built by NASA JPL. "
        "You are an AI that helps researchers analyze NASA's "
        "NEX-GDDP-CMIP6 (NASA Earth Exchange Global Daily Downscaled Projections, "
        "Coupled Model Intercomparison Project Phase 6) climate projection data "
        "(0.25 degree resolution, daily, 1950-2100). "
        "You can load data, process it, run analyses, and create visualizations.\n\n"

        "## IMPORTANT: Ask ONE question at a time\n"
        "When the user's request is missing key parameters, guide them step by step. "
        "Ask only ONE question per message and wait for the answer before asking the next. "
        "NEVER ask multiple questions in a single message.\n\n"

        "Format each question with a numbered list of options so the user can pick easily.\n\n"

        "When NO dataset is loaded yet, follow this order (skip steps the user already answered):\n"
        "Step 1 - Ask what they want to study (temperature trends, precipitation, extremes, etc.)\n"
        "Step 2 - Ask which variable:\n"
        "   1. tasmax - Daily Maximum Temperature\n"
        "   2. tasmin - Daily Minimum Temperature\n"
        "   3. pr - Precipitation\n"
        "   4. tas - Near-Surface Air Temperature\n"
        "   (list only the most relevant 4-6 options)\n"
        "Step 3 - Ask which climate model:\n"
        "   1. ACCESS-CM2 — Well-validated coupled model from CSIRO, Australia; strong in tropical and Southern Hemisphere climate\n"
        "   2. GFDL-ESM4 — NOAA's earth system model with detailed atmospheric chemistry; excels at air quality and carbon cycle studies\n"
        "   3. MRI-ESM2-0 — Japan Meteorological Agency model; known for reliable precipitation and typhoon representation\n"
        "   4. CanESM5 — Canadian Earth System Model; runs warm, useful for bracketing high-sensitivity climate responses\n"
        "   5. CESM2 — NCAR's Community Earth System Model; one of the most widely used and extensively documented models\n"
        "   (suggest these 5 popular models; if the user wants others, use list_available_models to show all 35)\n"
        "Step 4 - Ask about the scenario:\n"
        "   1. ssp245 - Middle of the road\n"
        "   2. ssp126 - Low emissions (sustainability)\n"
        "   3. ssp370 - High emissions\n"
        "   4. ssp585 - Very high emissions\n"
        "Step 5 - Ask about region (offer presets + custom):\n"
        "   1. California (32-42N, 124-114W)\n"
        "   2. Texas (25.5-36.5N, 106.5-93.5W)\n"
        "   3. Thailand\n"
        "   4. Global\n"
        "   5. Custom (I'll specify coordinates)\n"
        "Step 6 - Ask about time period (suggest decade ranges)\n\n"

        "You may skip steps if the user already specified those parameters. "
        "For example, if they say 'I want to look at temperature in California', "
        "you already know the variable (tasmax) and region, so skip to asking model.\n\n"

        "## CRITICAL: Always suggest next steps\n"
        "After EVERY completed action (loading data, generating a plot, running analysis, etc.), "
        "you MUST end your response with a numbered list of next-step options. "
        "This is required — never end a response without offering choices.\n\n"
        "After loading data:\n"
        "  **What would you like to do next?**\n"
        "  1. Show a spatial map\n"
        "  2. Plot the time series with trend\n"
        "  3. Calculate statistics\n"
        "  4. Analyze extreme heat events\n"
        "  5. Compare with another scenario\n\n"
        "After generating a visualization or analysis:\n"
        "  **What would you like to do next?**\n"
        "  1. Show a spatial map of the data\n"
        "  2. Calculate climate extreme indices (ETCCDI)\n"
        "  3. Analyze heatwave patterns\n"
        "  4. Compare with a different scenario or model\n"
        "  5. Load a different variable or region\n"
        "  6. Download the data\n\n"
        "Adapt the options based on what has already been done — don't repeat completed steps. "
        "Always present at least 3-4 relevant options.\n\n"

        "## Available Data\n"
        f"**Models** ({len(CMIP6_MODELS)} total): {models_str}, ...\n"
        f"**Variables**: {vars_str}\n"
        f"**Scenarios**:\n{scenarios_str}\n"
        "**Time range**: historical (1950-2014), SSP projections (2015-2100)\n"
        "**Resolution**: 0.25 degree (~25km)\n\n"

        "## Capabilities\n"
        "- **Data loading**: load_climate_data, load_multi_model_ensemble\n"
        "- **Processing**: spatial_subset, temporal_subset, regrid, convert_units, mask_by_country, "
        "calculate_anomaly, calculate_standardized_anomaly\n"
        "- **Analysis**: calculate_statistics, calculate_trend, calculate_climatology, "
        "calculate_regional_mean, analyze_heatwaves, calculate_batch_etccdi, calculate_correlation, "
        "calculate_eof, detect_extreme_events, calculate_ensemble_statistics, compare_scenarios, "
        "calculate_time_of_emergence\n"
        "- **Visualization**: generate_map, generate_timeseries_plot, generate_histogram, "
        "generate_country_map, generate_hovmoller, generate_scenario_fan_chart, generate_ensemble_spread_plot\n\n"

        "## Advanced workflow recipes (chain these for richer analyses)\n"
        "**Spatiotemporal anomaly detection** (find extreme heat/cold/wet/dry events):\n"
        "  load_climate_data -> calculate_standardized_anomaly(baseline=historical period) "
        "-> detect_extreme_events(sigma_threshold=2.0) -> generate_hovmoller (visualize)\n\n"
        "**Dominant variability patterns** (EOF/PCA):\n"
        "  load_climate_data -> calculate_anomaly -> calculate_eof(n_modes=3) "
        "-> generate_map(spatial_dataset_id) AND generate_timeseries_plot([pc_dataset_id])\n\n"
        "**Multi-model ensemble** (IPCC-style with model agreement):\n"
        "  load_multi_model_ensemble(models=[5 picks]) -> calculate_ensemble_statistics(baseline_start, baseline_end) "
        "-> generate_map(agreement_id, colormap='rdbu_r') AND generate_ensemble_spread_plot(ensemble_dataset_id)\n\n"
        "**Scenario comparison** (low vs high emissions):\n"
        "  compare_scenarios(scenarios=['ssp126','ssp245','ssp585']) "
        "-> generate_scenario_fan_chart(per_scenario mapping) AND generate_map(differences[i].dataset_id)\n\n"
        "**Time-of-emergence** (when does a cell experience climate change?):\n"
        "  load_climate_data(2015-2100, ssp585) -> calculate_time_of_emergence(baseline_start='1950-01-01', "
        "baseline_end='2014-12-31') -> generate_map(emergence_year, colormap='viridis')\n\n"
        "**Ensemble weighting** (which models to trust most for a region):\n"
        "  load_multi_model_ensemble(...) -> load_reference_dataset(source='ncep-reanalysis-monthly', ...) "
        "-> validate_ensemble_weighting(train='1980-2002', test='2003-2014') "
        "-> calculate_model_weights(method='combined') -> apply_ensemble_weights(weights) -> generate_map(...)\n\n"

        "## Python Code Execution\n"
        "You can write and execute Python code using `execute_python_code` for custom analyses "
        "that built-in tools cannot handle. Examples:\n"
        "- Custom statistical tests, curve fitting, regression\n"
        "- Complex data transformations or reshaping\n"
        "- Multi-dataset computations or custom comparisons\n"
        "- Custom visualizations beyond the standard plot types\n"
        "- Any analysis requiring custom logic\n\n"
        "Use `get_dataset('dataset_id')` to access loaded data, `store_dataset(data, 'desc')` to save results.\n"
        "**For downloads**: Use `url = save_to_csv(data)` or `url = save_to_netcdf(data)` — "
        "they return a download URL. Print the URL as a markdown link: `print(f'[Download CSV]({url})')`. "
        "Do NOT use `data.to_netcdf()` or `data.to_csv()` directly — they will fail on in-memory datasets.\n"
        "Libraries available: xarray (xr), numpy (np), pandas (pd), matplotlib.pyplot (plt), scipy.\n"
        "Always use `print()` to show results. Prefer built-in tools for standard operations.\n\n"

        "## Guidelines\n"
        "- When the user mentions a region by name, use appropriate lat/lon bounds\n"
        "- Chain tools as needed: load -> convert_units -> analyze -> visualize\n"
        "- Always convert temperature from Kelvin to Celsius (use convert_units with target_unit='degC')\n"
        "- Explain results in plain, accessible language\n"
        "- Common regions: California (32-42N, 124-114W), Texas (25.5-36.5N, 106.5-93.5W), "
        "Florida (24.5-31N, 87.5-80W), Global (-60-90N, -180-180E)\n\n"
        "## CRITICAL: Ambiguous country names\n"
        "Some country names are ambiguous (e.g., 'Korea' could be South Korea or North Korea, "
        "'Congo' could be Republic of the Congo or Democratic Republic of the Congo). "
        "If `get_country_bounds` returns an 'ambiguous' result with multiple matches, "
        "you MUST ask the user to clarify which country they mean before proceeding. "
        "NEVER assume one over the other — always ask.\n\n"

        "## CRITICAL: Country-specific data loading\n"
        "When the user specifies a COUNTRY (e.g., India, Thailand, Brazil):\n"
        "1. First call `get_country_bounds` to get the country's bounding box\n"
        "2. If the result contains 'ambiguous: true', ask the user to pick from the listed matches\n"
        "3. Then call `load_climate_data` using those lat/lon bounds (add ~1 degree buffer)\n"
        "4. Then call `mask_by_country` to clip the data precisely to the country's borders\n"
        "NEVER load global data when a country is specified. Always use the country's bounding box.\n"
        "This is critical for performance — global data is extremely large and slow to process.\n\n"

        "## Pushing results to MMGIS\n"
        "You CAN push climate analysis results directly into the live MMGIS map. "
        "Use this workflow when the user asks to 'push to MMGIS', 'show on map', or 'visualize in MMGIS':\n"
        "1. export_climate_geotiff(dataset_id) → returns cog_url\n"
        "2. push_layer_to_mmgis(layer_name, cog_url, layer_type='tile', colormap='rdbu_r') → returns browser_url\n"
        "Then tell the user to open the browser_url to see the layer in MMGIS.\n"
        "For vector/point data, use export_climate_geojson + layer_type='vector' instead.\n"
        "MMGIS is already running and connected — always attempt the push, never say you can't do it."
    )


def _extract_dataset_ids_from_args(args: dict) -> list[str]:
    """Extract input dataset IDs from tool arguments."""
    ids = []
    for key in ("dataset_id", "dataset1_id", "dataset2_id"):
        if key in args:
            ids.append(args[key])
    if "dataset_ids" in args:
        ids.extend(args["dataset_ids"])
    return ids


def _extract_dataset_ids_from_result(result: dict) -> list[str]:
    """Extract output dataset IDs from tool results."""
    ids = []
    if "dataset_id" in result and result["dataset_id"]:
        ids.append(result["dataset_id"])
    if "computed_indices" in result:
        ids.extend(result["computed_indices"].values())
    return ids


def _summarize_tool_result(tool_name: str, result: dict) -> str:
    """Create a short summary of a tool result for the DAG."""
    if "error" in result:
        return f"Error: {result['error'][:100]}"
    if tool_name == "calculate_statistics":
        mean = result.get("mean")
        return f"mean={mean:.2f}" if mean is not None else "computed"
    if tool_name == "calculate_trend":
        trend = result.get("trend", {})
        slope = trend.get("slope_per_decade")
        return f"slope={slope:.4f}/decade" if slope is not None else "computed"
    if tool_name in ("generate_map", "generate_timeseries_plot", "generate_histogram", "generate_country_map"):
        return "image generated" if result.get("image_base64") else "no image"
    if tool_name == "load_climate_data":
        dims = result.get("dimensions", {})
        return f"loaded {dims.get('time', '?')}t x {dims.get('lat', '?')}x{dims.get('lon', '?')}"
    if tool_name == "execute_python_code":
        stdout = result.get("stdout", "")
        n_images = len(result.get("images", []))
        parts = []
        if stdout:
            parts.append(stdout[:200])
        if n_images:
            parts.append(f"{n_images} figure{'s' if n_images > 1 else ''}")
        return " | ".join(parts) if parts else "code executed"
    if "dataset_id" in result:
        return f"dataset_id={result['dataset_id']}"
    return "completed"


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_chat(messages: list[dict], dataset_id: str | None):
    """Async generator that streams SSE events for a chat interaction."""
    client = _get_azure_client()
    if client is None:
        yield _sse_event("message_start", {"request_id": "no-client"})
        yield _sse_event("text_delta", {"content": "Azure OpenAI is not configured. Please set AZURE_OPENAI_ENDPOINT."})
        yield _sse_event("message_end", {"request_id": "no-client"})
        return

    request_id = str(uuid.uuid4())[:8]
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    tools = _get_chat_tools()

    context = ""
    if dataset_id:
        context = f"\n\nThe user currently has dataset '{dataset_id}' loaded."
    # List available datasets for context
    try:
        loaded = data_access.list_loaded_datasets()
        ds_list = loaded.get("datasets", [])
        if ds_list:
            context += "\n\nCurrently loaded datasets:\n"
            for ds_info in ds_list[:10]:
                context += f"- {ds_info.get('id')}: {ds_info.get('variable', '?')} / {ds_info.get('model', '?')} / {ds_info.get('scenario', '?')}\n"
    except Exception:
        pass

    full_messages = [
        {"role": "system", "content": _build_system_prompt() + context},
        *messages,
    ]

    yield _sse_event("message_start", {"request_id": request_id})

    try:
        while True:
            # Stream the response
            stream = client.chat.completions.create(
                model=deployment,
                messages=full_messages,
                tools=tools,
                stream=True,
            )

            collected_content = ""
            tool_calls_accum: dict[int, dict] = {}  # index -> {id, name, arguments}

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason  # noqa: F841 - reserved for future use

                # Stream text content
                if delta.content:
                    collected_content += delta.content
                    yield _sse_event("text_delta", {"content": delta.content})

                # Accumulate tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tool_calls_accum[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_accum[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_accum[idx]["arguments"] += tc_delta.function.arguments

            # If finish_reason is stop, we're done
            if not tool_calls_accum:
                break

            # Process tool calls
            if collected_content:
                full_messages.append({"role": "assistant", "content": collected_content, "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls_accum.values()
                ]})
            else:
                full_messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls_accum.values()
                ]})

            for tc in tool_calls_accum.values():
                tool_call_id = tc["id"]
                tool_name = tc["name"]
                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    args = {}

                # Normalize common LLM argument mistakes
                args = _normalize_tool_args(tool_name, args)

                input_ids = _extract_dataset_ids_from_args(args)
                yield _sse_event("tool_start", {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "args": args,
                    "inputs": input_ids,
                })

                impl = _CHAT_TOOL_IMPLS.get(tool_name)
                start_time = time.time()

                if impl is None:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                    yield _sse_event("tool_error", {
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "error": tool_result["error"],
                        "duration_ms": 0,
                    })
                else:
                    try:
                        # Run ALL tools with progress streaming support.
                        # Tools that call the thread-local progress_callback
                        # will have their updates streamed to the UI in real time.
                        from rcmes_mcp.tools.data_access import _thread_local as _da_tl
                        loop = asyncio.get_running_loop()
                        progress_queue: asyncio.Queue = asyncio.Queue()

                        def _progress_cb(completed: int, total: int, detail: str):
                            loop.call_soon_threadsafe(
                                progress_queue.put_nowait,
                                {"completed": completed, "total": total, "detail": detail},
                            )

                        def _run_with_progress():
                            _da_tl.progress_callback = _progress_cb
                            try:
                                return impl(**args)
                            finally:
                                _da_tl.progress_callback = None

                        task = asyncio.ensure_future(asyncio.to_thread(_run_with_progress))

                        while not task.done():
                            try:
                                item = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                                yield _sse_event("tool_progress", {
                                    "tool_call_id": tool_call_id,
                                    "progress": item,
                                })
                            except asyncio.TimeoutError:
                                continue

                        tool_result = task.result()

                        # Drain remaining progress events
                        while not progress_queue.empty():
                            item = progress_queue.get_nowait()
                            yield _sse_event("tool_progress", {
                                "tool_call_id": tool_call_id,
                                "progress": item,
                            })

                        duration_ms = int((time.time() - start_time) * 1000)

                        if "error" in tool_result:
                            yield _sse_event("tool_error", {
                                "tool_call_id": tool_call_id,
                                "tool_name": tool_name,
                                "error": tool_result["error"],
                                "duration_ms": duration_ms,
                            })
                        else:
                            # Emit image(s) separately if present
                            if tool_result.get("image_base64"):
                                image_url = _save_image(tool_result["image_base64"])
                                yield _sse_event("image", {
                                    "tool_call_id": tool_call_id,
                                    "image_base64": tool_result["image_base64"],
                                    "image_url": image_url,
                                })
                            for img in tool_result.get("images", []):
                                image_url = _save_image(img)
                                yield _sse_event("image", {
                                    "tool_call_id": tool_call_id,
                                    "image_base64": img,
                                    "image_url": image_url,
                                })

                            output_ids = _extract_dataset_ids_from_result(tool_result)
                            summary = _summarize_tool_result(tool_name, tool_result)
                            yield _sse_event("tool_complete", {
                                "tool_call_id": tool_call_id,
                                "tool_name": tool_name,
                                "result_summary": summary,
                                "outputs": output_ids,
                                "duration_ms": duration_ms,
                                "status": "success",
                            })
                    except Exception as e:
                        duration_ms = int((time.time() - start_time) * 1000)
                        tool_result = {"error": str(e)}
                        yield _sse_event("tool_error", {
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "error": str(e),
                            "duration_ms": duration_ms,
                        })

                # Strip large fields before sending back to LLM
                tool_result_for_llm = {
                    k: v for k, v in tool_result.items()
                    if k not in ("image_base64", "images")
                }
                if "image_base64" in tool_result:
                    tool_result_for_llm["image_generated"] = True
                if tool_result.get("images"):
                    tool_result_for_llm["images_generated"] = len(tool_result["images"])

                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result_for_llm, default=str),
                })

            # Continue the loop to get the next response

    except Exception as e:
        logger.exception("Streaming chat error")
        yield _sse_event("text_delta", {"content": f"\n\nError: {e}"})

    yield _sse_event("message_end", {"request_id": request_id})


@app.post("/api/chat/stream")
async def chat_stream(request: ChatStreamRequest):
    """Stream a chat response as Server-Sent Events."""
    return StreamingResponse(
        _stream_chat(request.messages, request.dataset_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _azure_chat(message: str, dataset_id: str | None) -> dict[str, Any]:
    """Handle chat via Azure OpenAI with function calling."""
    client = _get_azure_client()
    if client is None:
        return None  # type: ignore[return-value]

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    tools = _get_chat_tools()

    context = ""
    if dataset_id:
        context = f"\n\nThe user currently has dataset '{dataset_id}' loaded."

    messages = [
        {"role": "system", "content": _build_system_prompt() + context},
        {"role": "user", "content": message},
    ]

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=tools,
        )

        choice = response.choices[0]
        action_result = None

        # Process tool calls (single round)
        if choice.finish_reason == "tool_calls":
            assistant_message = choice.message
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                args = _normalize_tool_args(tool_name, args)
                impl = _CHAT_TOOL_IMPLS.get(tool_name)
                if impl is None:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        tool_result = impl(**args)
                    except Exception as e:
                        tool_result = {"error": str(e)}

                # Capture visualization results for the frontend
                if tool_result.get("image_base64"):
                    action_result = {
                        "image_base64": tool_result["image_base64"],
                        "message": f"Generated {tool_name.replace('generate_', '')} visualization.",
                    }
                elif "error" not in tool_result:
                    action_result = {
                        "dataset_id": tool_result.get("dataset_id"),
                        "message": f"Tool `{tool_name}` completed successfully.",
                    }

                # Strip large fields (base64 images) before sending back to LLM
                tool_result_for_llm = {
                    k: v for k, v in tool_result.items()
                    if k != "image_base64"
                }
                if "image_base64" in tool_result:
                    tool_result_for_llm["image_generated"] = True

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result_for_llm, default=str),
                    }
                )

            # Get the follow-up response
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                tools=tools,
            )
            choice = response.choices[0]

        response_text = choice.message.content or ""
        return {"response": response_text, "action_result": action_result}

    except Exception as e:
        logger.exception("Azure OpenAI chat error")
        return {"response": f"Error communicating with Azure OpenAI: {e}", "action_result": None}


def _keyword_chat(message: str, dataset_id: str | None) -> dict[str, Any]:
    """Fallback keyword-based chat when Azure OpenAI is not configured."""
    msg = message.lower()
    response_text = ""
    action_result = None

    def fmt(value, decimals=2):
        if value is None:
            return "N/A"
        try:
            return f"{value:.{decimals}f}"
        except (TypeError, ValueError):
            return str(value)

    if any(word in msg for word in ["statistics", "stats", "summary"]):
        if dataset_id:
            response_text = "I'll calculate the statistics for your current dataset."
            try:
                result = analysis.calculate_statistics(dataset_id=dataset_id)
                if "error" not in result:
                    action_result = {
                        "dataset_id": result.get("dataset_id"),
                        "message": (
                            f"**Statistics:**\n"
                            f"- Mean: {fmt(result.get('mean'))}\n"
                            f"- Std: {fmt(result.get('std'))}\n"
                            f"- Min: {fmt(result.get('min'))}\n"
                            f"- Max: {fmt(result.get('max'))}"
                        ),
                    }
                else:
                    response_text = f"Statistics error: {result.get('error')}"
            except Exception as e:
                response_text = f"Error calculating statistics: {e}"
        else:
            response_text = "Please load a dataset first before requesting statistics."

    elif any(word in msg for word in ["trend", "change", "increase", "decrease"]):
        if dataset_id:
            response_text = "I'll analyze the trend in your data."
            try:
                result = analysis.calculate_trend(dataset_id=dataset_id)
                if "error" not in result:
                    trend = result.get("trend", {})
                    slope = trend.get("slope_per_decade")
                    if slope is not None:
                        direction = "increasing" if slope > 0 else "decreasing"
                        action_result = {
                            "dataset_id": result.get("dataset_id"),
                            "message": f"**Trend Analysis:**\nThe data shows a {direction} trend of {abs(slope):.4f} per decade.",
                        }
                    else:
                        response_text = "Trend analysis completed but no slope value was returned."
                else:
                    response_text = f"Trend error: {result.get('error')}"
            except Exception as e:
                response_text = f"Error analyzing trend: {e}"
        else:
            response_text = "Please load a dataset first to analyze trends."

    elif any(word in msg for word in ["heatwave", "heat wave", "extreme", "hot"]):
        if dataset_id:
            response_text = "I'll perform a heatwave analysis on your data."
            try:
                result = indices.analyze_heatwaves(dataset_id=dataset_id)
                if "error" not in result:
                    summary = result.get("summary", {})
                    hot_days = summary.get("mean_annual_hot_days")
                    heatwave_freq = summary.get("mean_annual_heatwave_frequency")
                    action_result = {
                        "dataset_id": result.get("dataset_id"),
                        "message": (
                            f"**Heatwave Analysis:**\n"
                            f"- Hot days per year: {fmt(hot_days, 1)}\n"
                            f"- Heatwave events: {fmt(heatwave_freq, 1)}/year"
                        ),
                    }
                else:
                    response_text = f"Heatwave error: {result.get('error')}"
            except Exception as e:
                response_text = f"Error with heatwave analysis: {e}"
        else:
            response_text = "Please load temperature data first (tasmax recommended)."

    elif any(word in msg for word in ["map", "spatial", "plot map"]):
        if dataset_id:
            response_text = "I'll generate a spatial map for you."
            try:
                result = visualization.generate_map(dataset_id=dataset_id, title="Climate Data Map")
                if "error" not in result and result.get("image_base64"):
                    action_result = {
                        "image_base64": result["image_base64"],
                        "message": "Here's the spatial map of your data.",
                    }
            except Exception as e:
                response_text = f"Error generating map: {e}"
        else:
            response_text = "Please load a dataset first to generate a map."

    elif any(word in msg for word in ["timeseries", "time series", "temporal", "over time"]):
        if dataset_id:
            response_text = "I'll generate a time series plot for you."
            try:
                result = visualization.generate_timeseries_plot(
                    dataset_ids=[dataset_id], title="Climate Time Series", show_trend=True
                )
                if "error" not in result and result.get("image_base64"):
                    action_result = {
                        "image_base64": result["image_base64"],
                        "message": "Here's the time series plot.",
                    }
            except Exception as e:
                response_text = f"Error generating time series: {e}"
        else:
            response_text = "Please load a dataset first to generate a time series plot."

    elif any(word in msg for word in ["compare", "scenario", "ssp"]):
        response_text = (
            "To compare different scenarios:\n"
            "1. Load data for each scenario you want to compare\n"
            "2. Use the visualization tools to overlay them\n"
            "3. Run trend analysis on each to quantify differences\n\n"
            "SSP scenarios:\n"
            "- **SSP1-2.6**: Sustainable development, low emissions\n"
            "- **SSP2-4.5**: Middle of the road\n"
            "- **SSP3-7.0**: Regional rivalry, high emissions\n"
            "- **SSP5-8.5**: Fossil-fuel intensive, very high emissions"
        )

    elif any(word in msg for word in ["help", "what can", "how do"]):
        response_text = (
            "I can help you with climate data analysis!\n\n"
            "**Data Analysis:** statistics, trends, heatwaves\n"
            "**Visualizations:** maps, time series, histograms\n\n"
            "Load data using the sidebar, then ask me questions!"
        )

    elif any(word in msg for word in ["hello", "hi", "hey"]):
        response_text = (
            "Hello! I'm your climate data assistant. "
            "Load some data using the sidebar controls, then ask me anything!"
        )

    else:
        if dataset_id:
            response_text = (
                "Try asking me to: show statistics, analyze the trend, "
                "generate a map, create a time series plot, or detect heatwaves."
            )
        else:
            response_text = (
                "Please load climate data using the sidebar controls first, "
                "then I can help you analyze and visualize it!"
            )

    return {"response": response_text, "action_result": action_result}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    """Process a chat message using Azure OpenAI (with keyword fallback)."""
    # Try Azure OpenAI first
    result = await _azure_chat(request.message, request.dataset_id)
    if result is not None:
        return result

    # Fall back to keyword-based responses
    return _keyword_chat(request.message, request.dataset_id)


# ============================================================================
# Health Check
# ============================================================================


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint with session stats."""
    datasets = session_manager.list_datasets()
    return {
        "status": "healthy",
        "service": "rcmes-api",
        "datasets_loaded": len(datasets),
    }


# ============================================================================
# Temporary Download Links
# ============================================================================


@app.get("/api/downloads/{token}/{filename}")
async def serve_download(token: str, filename: str):
    """Serve a temporary download file by token."""
    _cleanup_expired_downloads()

    info = _download_registry.get(token)
    if info is None:
        raise HTTPException(status_code=404, detail="Download link expired or not found")

    filepath = Path(info["filepath"])
    if not filepath.exists():
        _download_registry.pop(token, None)
        raise HTTPException(status_code=404, detail="File no longer available")

    return FileResponse(
        filepath,
        media_type=info["media_type"],
        filename=info["filename"],
        headers={"Content-Disposition": f'attachment; filename="{info["filename"]}"'},
    )


@app.get("/api/download-dataset/{dataset_id}")
async def download_dataset_direct(dataset_id: str, format: str = "csv"):
    """Generate a temporary download link for a loaded dataset.

    Returns a JSON object with a download URL that can be shared.
    """
    try:
        ds = session_manager.get(dataset_id)
        metadata = session_manager.get_metadata(dataset_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    from rcmes_mcp.utils.validation import check_download_size
    try:
        check_download_size(ds)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))

    if hasattr(ds, 'compute'):
        ds = ds.compute()

    var_name = metadata.variable or "data"
    model_name = metadata.model or "model"

    if format == "netcdf":
        filename = f"{var_name}_{model_name}_{dataset_id}.nc"
        filepath = DOWNLOADS_DIR / filename
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset(name=var_name)
        ds.to_netcdf(filepath, engine='scipy')
        media_type = "application/x-netcdf"
    elif format == "csv":
        filename = f"{var_name}_{model_name}_{dataset_id}.csv"
        filepath = DOWNLOADS_DIR / filename
        if isinstance(ds, xr.DataArray):
            df = ds.to_dataframe().reset_index()
        else:
            df = ds.to_dataframe().reset_index()
        df.to_csv(filepath, index=False)
        media_type = "text/csv"
    else:
        raise HTTPException(status_code=400, detail="Format must be 'netcdf' or 'csv'")

    url = _create_download_link(filepath, filename, media_type)

    return {
        "success": True,
        "download_url": url,
        "filename": filename,
        "format": format,
        "expires_in_hours": _DOWNLOAD_TTL_HOURS,
    }


# ============================================================================
# Persisted Chat Images
# ============================================================================

@app.get("/api/images/{filename}")
async def serve_image(filename: str):
    """Serve a persisted chat image."""
    filepath = IMAGES_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        filepath,
        media_type="image/png",
        headers={"Cache-Control": "no-cache, max-age=0"},
    )


# ============================================================================
# MMGIS Layer File Serving
# Serves COG and GeoJSON files written by export_climate_geotiff /
# export_climate_geojson so that TiTiler and MMGIS can fetch them.
# ============================================================================

_MMGIS_DATA_DIR = Path(os.environ.get("MMGIS_DATA_DIR", str(Path.home() / ".rcmes" / "layers")))


@app.get("/files/{filename}")
async def serve_layer_file(filename: str):
    """Serve a COG or GeoJSON layer file for MMGIS / TiTiler consumption."""
    # Prevent path traversal
    safe_name = Path(filename).name
    filepath = _MMGIS_DATA_DIR / safe_name
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"Layer file '{safe_name}' not found")
    media_type = (
        "application/geo+json" if safe_name.endswith(".geojson")
        else "image/tiff"
    )
    return FileResponse(
        filepath,
        media_type=media_type,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600",
        },
    )


# ============================================================================
# MMGIS Integration Endpoints
# ============================================================================

class ExportGeoTIFFRequest(BaseModel):
    dataset_id: str
    variable: str | None = None
    time_aggregation: str = "mean"
    time_index: int | None = None
    output_filename: str | None = None


class ExportGeoJSONRequest(BaseModel):
    dataset_id: str
    variable: str | None = None
    statistic: str = "mean"
    output_filename: str | None = None


class PushToMMGISRequest(BaseModel):
    layer_name: str
    data_url: str
    layer_type: str = "tile"
    colormap: str = "rdbu_r"
    description: str = ""
    opacity: float = 0.8


@app.post("/api/export-geotiff")
async def api_export_geotiff(req: ExportGeoTIFFRequest):
    """Export a session dataset to a Cloud-Optimized GeoTIFF served by TiTiler."""
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: mmgis.export_climate_geotiff(
            dataset_id=req.dataset_id,
            variable=req.variable,
            time_aggregation=req.time_aggregation,
            time_index=req.time_index,
            output_filename=req.output_filename,
        ),
    )
    if "error" in result.get("status", ""):
        raise HTTPException(status_code=500, detail=result)
    return result


@app.post("/api/export-geojson")
async def api_export_geojson(req: ExportGeoJSONRequest):
    """Export a session dataset to a GeoJSON feature collection."""
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: mmgis.export_climate_geojson(
            dataset_id=req.dataset_id,
            variable=req.variable,
            statistic=req.statistic,
            output_filename=req.output_filename,
        ),
    )
    if "error" in result.get("status", ""):
        raise HTTPException(status_code=500, detail=result)
    return result


@app.post("/api/push-to-mmgis")
async def api_push_to_mmgis(req: PushToMMGISRequest):
    """Push a COG or GeoJSON layer into MMGIS as a live map layer."""
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: mmgis.push_layer_to_mmgis(
            layer_name=req.layer_name,
            data_url=req.data_url,
            layer_type=req.layer_type,
            colormap=req.colormap,
            description=req.description,
            opacity=req.opacity,
        ),
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result)
    return result


# Forwards /mmgis/* → http://localhost:2888/* so MMGIS is reachable through
# the already-open port 8502 without needing a separate firewall rule.
# ============================================================================

def _get_mmgis_internal_url() -> str:
    return os.environ.get("MMGIS_URL", "http://127.0.0.1:2888/mmgis")


@app.api_route(
    "/mmgis/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def mmgis_proxy(path: str, request: Request):
    """Transparent reverse proxy to the internal MMGIS container."""
    import httpx as _httpx

    _base = _get_mmgis_internal_url()
    target = f"{_base.rstrip('/')}/{path}"
    qs = request.url.query
    if qs:
        target = f"{target}?{qs}"

    # Forward the body for non-GET requests
    body = await request.body()

    # Strip hop-by-hop headers; rewrite Host
    skip = {"host", "connection", "transfer-encoding", "te", "trailers", "upgrade"}
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
    fwd_headers["host"] = _base.split("//", 1)[-1].split("/")[0]
    # Request uncompressed so the proxy can forward raw bytes without encoding mismatch
    fwd_headers["accept-encoding"] = "identity"

    async with _httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        resp = await client.request(
            method=request.method,
            url=target,
            headers=fwd_headers,
            content=body,
        )

    # Rewrite Location headers so redirects stay within the proxy
    resp_headers = dict(resp.headers)
    resp_headers.pop("transfer-encoding", None)
    resp_headers.pop("content-encoding", None)
    if "location" in resp_headers:
        loc = resp_headers["location"]
        mmgis_base = _base.rstrip("/")
        if loc.startswith(mmgis_base):
            resp_headers["location"] = loc[len(mmgis_base):]

    from fastapi.responses import Response as _Resp
    return _Resp(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
    )


# Forwards /titiler/* → http://localhost:8080/* so TiTiler is reachable through
# the already-open port 8502 without needing a separate firewall rule.

@app.api_route(
    "/titiler/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def titiler_proxy(path: str, request: Request):
    """Transparent reverse proxy to the internal TiTiler container."""
    import httpx as _httpx

    _titiler_base = os.environ.get("TITILER_INTERNAL_URL", "http://127.0.0.1:8080")
    target = f"{_titiler_base.rstrip('/')}/{path}"
    qs = request.url.query
    if qs:
        target = f"{target}?{qs}"

    body = await request.body()
    skip = {"host", "connection", "transfer-encoding", "te", "trailers", "upgrade"}
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
    fwd_headers["host"] = _titiler_base.split("//", 1)[-1].split("/")[0]
    fwd_headers["accept-encoding"] = "identity"

    async with _httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        resp = await client.request(
            method=request.method,
            url=target,
            headers=fwd_headers,
            content=body,
        )

    resp_headers = dict(resp.headers)
    resp_headers.pop("transfer-encoding", None)
    resp_headers.pop("content-encoding", None)

    from fastapi.responses import Response as _Resp
    return _Resp(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
    )


if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API routes."""
        # Try to serve the requested file
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Fall back to index.html for client-side routing
        # Always serve index.html with no-cache so browsers pick up new asset hashes
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )


def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8502)


if __name__ == "__main__":
    main()
