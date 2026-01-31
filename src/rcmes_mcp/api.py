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

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Import RCMES tools
from rcmes_mcp.tools import analysis, data_access, indices, processing, visualization

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
    dataset_id: str = Field(..., description="Dataset ID")
    viz_type: str = Field(..., description="Type: map, timeseries, histogram")
    title: str | None = Field(None, description="Plot title")
    show_trend: bool = Field(False, description="Show trend line (for timeseries)")


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
        index_name=request.index_name,
        threshold=request.threshold,
        base_period_start=request.base_period_start,
        base_period_end=request.base_period_end,
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

    if viz_type == "map":
        result = visualization.generate_map(
            dataset_id=request.dataset_id,
            title=request.title,
        )
    elif viz_type == "timeseries":
        result = visualization.generate_timeseries_plot(
            dataset_ids=[request.dataset_id],
            title=request.title,
            show_trend=request.show_trend,
        )
    elif viz_type == "histogram":
        result = visualization.generate_histogram(
            dataset_id=request.dataset_id,
            title=request.title,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown visualization type: {viz_type}. Valid types: map, timeseries, histogram",
        )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ============================================================================
# Chat Endpoint
# ============================================================================


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    """Process a chat message and return AI response with optional actions."""
    message = request.message.lower()
    dataset_id = request.dataset_id

    # Simple keyword-based responses for demo
    # In production, this would connect to an LLM
    response_text = ""
    action_result = None

    # Helper function to safely format numbers
    def fmt(value, decimals=2):
        if value is None:
            return "N/A"
        try:
            return f"{value:.{decimals}f}"
        except (TypeError, ValueError):
            return str(value)

    # Check for analysis requests
    if any(word in message for word in ["statistics", "stats", "summary"]):
        if dataset_id:
            response_text = "I'll calculate the statistics for your current dataset."
            try:
                result = analysis.calculate_statistics(dataset_id=dataset_id)
                if "error" not in result:
                    action_result = {
                        "dataset_id": result.get("dataset_id"),
                        "message": f"**Statistics:**\n- Mean: {fmt(result.get('mean'))}\n- Std: {fmt(result.get('std'))}\n- Min: {fmt(result.get('min'))}\n- Max: {fmt(result.get('max'))}",
                    }
                else:
                    response_text = f"Statistics calculation returned an error: {result.get('error', 'Unknown error')}"
            except Exception as e:
                response_text = f"I encountered an error calculating statistics: {str(e)}"
        else:
            response_text = "Please load a dataset first before requesting statistics. Use the sidebar controls to load climate data."

    elif any(word in message for word in ["trend", "change", "increase", "decrease"]):
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
                    response_text = f"Trend analysis returned an error: {result.get('error', 'Unknown error')}"
            except Exception as e:
                response_text = f"I encountered an error analyzing the trend: {str(e)}"
        else:
            response_text = "Please load a dataset first to analyze trends."

    elif any(word in message for word in ["heatwave", "heat wave", "extreme", "hot"]):
        if dataset_id:
            response_text = "I'll perform a heatwave analysis on your data."
            try:
                result = indices.analyze_heatwaves(dataset_id=dataset_id)
                if "error" not in result:
                    summary = result.get("summary", {})
                    hot_days = summary.get('mean_annual_hot_days')
                    heatwave_freq = summary.get('mean_annual_heatwave_frequency')
                    hot_days_str = f"{hot_days:.1f}" if hot_days is not None else "N/A"
                    heatwave_freq_str = f"{heatwave_freq:.1f}" if heatwave_freq is not None else "N/A"
                    action_result = {
                        "dataset_id": result.get("dataset_id"),
                        "message": f"**Heatwave Analysis:**\n- Hot days per year: {hot_days_str}\n- Heatwave events: {heatwave_freq_str}/year",
                    }
                else:
                    response_text = f"Heatwave analysis returned an error: {result.get('error', 'Unknown error')}"
            except Exception as e:
                response_text = f"I encountered an error with heatwave analysis: {str(e)}"
        else:
            response_text = "Please load temperature data first (tasmax is recommended for heatwave analysis)."

    elif any(word in message for word in ["map", "spatial", "plot map"]):
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
                response_text = f"I encountered an error generating the map: {str(e)}"
        else:
            response_text = "Please load a dataset first to generate a map visualization."

    elif any(word in message for word in ["timeseries", "time series", "temporal", "over time"]):
        if dataset_id:
            response_text = "I'll generate a time series plot for you."
            try:
                result = visualization.generate_timeseries_plot(
                    dataset_ids=[dataset_id],
                    title="Climate Time Series",
                    show_trend=True,
                )
                if "error" not in result and result.get("image_base64"):
                    action_result = {
                        "image_base64": result["image_base64"],
                        "message": "Here's the time series plot showing how the data changes over time.",
                    }
            except Exception as e:
                response_text = f"I encountered an error generating the time series: {str(e)}"
        else:
            response_text = "Please load a dataset first to generate a time series plot."

    elif any(word in message for word in ["compare", "scenario", "ssp"]):
        response_text = (
            "To compare different scenarios:\n"
            "1. Load data for each scenario you want to compare (e.g., SSP2-4.5 and SSP5-8.5)\n"
            "2. Use the visualization tools to overlay them\n"
            "3. Run trend analysis on each to quantify differences\n\n"
            "SSP scenarios represent different emission pathways:\n"
            "- **SSP1-2.6**: Sustainable development, low emissions\n"
            "- **SSP2-4.5**: Middle of the road\n"
            "- **SSP3-7.0**: Regional rivalry, high emissions\n"
            "- **SSP5-8.5**: Fossil-fuel intensive, very high emissions"
        )

    elif any(word in message for word in ["help", "what can", "how do"]):
        response_text = (
            "I can help you with climate data analysis! Here's what I can do:\n\n"
            "**Data Analysis:**\n"
            "- Calculate statistics (mean, std, percentiles)\n"
            "- Analyze temperature trends\n"
            "- Detect heatwaves and extreme events\n\n"
            "**Visualizations:**\n"
            "- Generate spatial maps\n"
            "- Create time series plots\n"
            "- Compare different scenarios\n\n"
            "**To get started:**\n"
            "1. Use the sidebar to select model, variable, and region\n"
            "2. Click 'Load Data' to fetch the climate data\n"
            "3. Ask me questions or request analyses!"
        )

    elif any(word in message for word in ["hello", "hi", "hey"]):
        response_text = (
            "Hello! I'm your climate data assistant. I can help you:\n"
            "- Analyze climate model projections\n"
            "- Calculate statistics and trends\n"
            "- Generate visualizations\n"
            "- Understand climate scenarios\n\n"
            "Load some data using the sidebar controls, then ask me anything!"
        )

    else:
        if dataset_id:
            response_text = (
                "I can help you analyze your loaded dataset. Try asking me to:\n"
                "- Show statistics\n"
                "- Analyze the trend\n"
                "- Generate a map\n"
                "- Create a time series plot\n"
                "- Detect heatwaves (for temperature data)"
            )
        else:
            response_text = (
                "To get started, please load some climate data using the controls in the sidebar. "
                "Select a model, variable, region, and time period, then click 'Load Data'. "
                "Once loaded, I can help you analyze and visualize the data!"
            )

    return {"response": response_text, "action_result": action_result}


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
