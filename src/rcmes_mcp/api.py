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

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import io
import xarray as xr

from rcmes_mcp.utils.session import session_manager

# Import RCMES tools
from rcmes_mcp.tools import analysis, data_access, indices, processing, visualization

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RCMES Climate API",
    description="REST API for NASA's NEX-GDDP-CMIP6 climate data analysis",
    version="0.1.0",
)

# Path to React build (web/dist after npm run build)
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

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


# ============================================================================
# Request/Response Models
# ============================================================================


class LoadDataRequest(BaseModel):
    variable: str = Field(..., description="Climate variable (e.g., tasmax, pr)")
    model: str = Field(..., description="Climate model name")
    scenario: str = Field(..., description="Emissions scenario (e.g., ssp585)")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    lat_min: float = Field(..., ge=-90, le=90)
    lat_max: float = Field(..., ge=-90, le=90)
    lon_min: float = Field(..., ge=-180, le=180)
    lon_max: float = Field(..., ge=-180, le=180)


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
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    start_date: str | None = None
    end_date: str | None = None


class RegridRequest(BaseModel):
    dataset_id: str
    target_resolution: float = Field(..., description="Target resolution in degrees")
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
    indices: list[str] = Field(..., description="List of ETCCDI index names (e.g., ['TXx', 'SU'])")
    freq: str = Field("YS", description="Output frequency: YS (annual), QS-DEC (seasonal), MS (monthly)")


class CountryMapRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID to visualize")
    country_name: str | None = Field(None, description="Country to highlight")
    title: str | None = Field(None, description="Plot title")
    colormap: str = Field("viridis", description="Matplotlib colormap name")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
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
    print(f"[visualize] viz_type={viz_type}, dataset_ids={ids}, labels={request.labels}")
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
                "name": "calculate_statistics",
                "description": "Calculate summary statistics (mean, std, min, max) for a loaded dataset",
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
                "description": "Calculate temporal trend with statistical significance",
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
                "description": "Generate a spatial map visualization",
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
                "description": "Generate a time series plot",
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
                "name": "calculate_batch_etccdi",
                "description": "Calculate multiple ETCCDI climate extreme indices at once",
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
    ]


# Map tool names to their implementations for the chat endpoint
_CHAT_TOOL_IMPLS: dict[str, Any] = {
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
    "calculate_batch_etccdi": indices.calculate_batch_etccdi,
    "generate_country_map": visualization.generate_country_map,
}

_CHAT_SYSTEM_PROMPT = (
    "You are a climate research assistant embedded in a web UI. "
    "The user may have a dataset already loaded (identified by a dataset_id). "
    "You have access to tools for statistics, trends, heatwave analysis, and visualization. "
    "When the user asks for analysis or plots, call the appropriate tool with the dataset_id. "
    "Always explain results in accessible language."
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
        {"role": "system", "content": _CHAT_SYSTEM_PROMPT + context},
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

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, default=str),
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
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "rcmes-api"}


# ============================================================================
# Static File Serving (React Frontend)
# ============================================================================

# Serve React app static files if built
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
        return FileResponse(STATIC_DIR / "index.html")


def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8502)


if __name__ == "__main__":
    main()
