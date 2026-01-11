"""
Gemini API Client for RCMES

This module provides a Gemini-compatible interface to RCMES tools,
bypassing the MCP protocol for direct function calling.

Usage:
    python -m rcmes_mcp.gemini_client

Requires:
    - GEMINI_API_KEY environment variable (or in .bashrc)
    - pip install google-generativeai
"""

from __future__ import annotations

import json
import os
from typing import Any

# Import RCMES tools directly
from rcmes_mcp.tools import data_access, processing, analysis, indices, visualization
from rcmes_mcp.utils.session import session_manager

# Map of available tools
RCMES_TOOLS = {
    # Data Access
    "list_available_models": data_access.list_available_models,
    "list_available_variables": data_access.list_available_variables,
    "list_available_scenarios": data_access.list_available_scenarios,
    "get_dataset_metadata": data_access.get_dataset_metadata,
    "load_climate_data": data_access.load_climate_data,
    "list_loaded_datasets": data_access.list_loaded_datasets,
    "get_dataset_info": data_access.get_dataset_info,
    "delete_dataset": data_access.delete_dataset,
    # Processing
    "temporal_subset": processing.temporal_subset,
    "spatial_subset": processing.spatial_subset,
    "temporal_resample": processing.temporal_resample,
    "convert_units": processing.convert_units,
    "regrid": processing.regrid,
    "calculate_anomaly": processing.calculate_anomaly,
    # Analysis
    "calculate_statistics": analysis.calculate_statistics,
    "calculate_climatology": analysis.calculate_climatology,
    "calculate_trend": analysis.calculate_trend,
    "calculate_regional_mean": analysis.calculate_regional_mean,
    "calculate_bias": analysis.calculate_bias,
    "calculate_correlation": analysis.calculate_correlation,
    "calculate_rmse": analysis.calculate_rmse,
    # Indices
    "list_climate_indices": indices.list_climate_indices,
    "calculate_etccdi_index": indices.calculate_etccdi_index,
    "analyze_heatwaves": indices.analyze_heatwaves,
    "calculate_drought_index": indices.calculate_drought_index,
    "calculate_growing_degree_days": indices.calculate_growing_degree_days,
    # Visualization
    "generate_map": visualization.generate_map,
    "generate_timeseries_plot": visualization.generate_timeseries_plot,
    "generate_comparison_map": visualization.generate_comparison_map,
    "generate_taylor_diagram": visualization.generate_taylor_diagram,
    "generate_histogram": visualization.generate_histogram,
}


def get_tool_descriptions() -> list[dict]:
    """Get tool descriptions for Gemini function calling."""
    return [
        {
            "name": "list_available_models",
            "description": "List available CMIP6 climate models in NEX-GDDP-CMIP6 dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "default": "NEX-GDDP-CMIP6"},
                    "scenario": {"type": "string", "description": "Filter by scenario (ssp126, ssp245, ssp370, ssp585)"},
                },
            },
        },
        {
            "name": "list_available_variables",
            "description": "List climate variables (tas, tasmax, pr, etc.) available in the dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "default": "NEX-GDDP-CMIP6"},
                },
            },
        },
        {
            "name": "load_climate_data",
            "description": "Load climate data for a specific region and time period from NEX-GDDP-CMIP6",
            "parameters": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string", "description": "Climate variable (tas, tasmax, tasmin, pr)"},
                    "model": {"type": "string", "description": "Climate model name (e.g., ACCESS-CM2, CESM2)"},
                    "scenario": {"type": "string", "description": "Emissions scenario (historical, ssp126, ssp245, ssp370, ssp585)"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                    "lat_min": {"type": "number", "description": "Minimum latitude"},
                    "lat_max": {"type": "number", "description": "Maximum latitude"},
                    "lon_min": {"type": "number", "description": "Minimum longitude"},
                    "lon_max": {"type": "number", "description": "Maximum longitude"},
                },
                "required": ["variable", "model", "scenario", "start_date", "end_date", "lat_min", "lat_max", "lon_min", "lon_max"],
            },
        },
        {
            "name": "convert_units",
            "description": "Convert dataset units (K to degC, kg/m2/s to mm/day)",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "ID of the dataset"},
                    "target_unit": {"type": "string", "description": "Target unit (degC, degF, mm/day)"},
                },
                "required": ["dataset_id", "target_unit"],
            },
        },
        {
            "name": "analyze_heatwaves",
            "description": "Analyze heatwave frequency, duration, and intensity from temperature data",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "ID of tasmax dataset"},
                    "threshold_percentile": {"type": "number", "default": 90},
                    "min_duration": {"type": "integer", "default": 3},
                },
                "required": ["dataset_id"],
            },
        },
        {
            "name": "calculate_trend",
            "description": "Calculate temporal trend with statistical significance",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "ID of the dataset"},
                },
                "required": ["dataset_id"],
            },
        },
        {
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
    ]


def execute_tool(tool_name: str, args: dict[str, Any]) -> dict:
    """Execute an RCMES tool and return the result."""
    if tool_name not in RCMES_TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = RCMES_TOOLS[tool_name](**args)
        return result
    except Exception as e:
        return {"error": str(e)}


def create_gemini_model():
    """Create a Gemini model with RCMES tools."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Please install google-generativeai: pip install google-generativeai")

    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")

    genai.configure(api_key=api_key)

    # Create model with function calling
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction="""You are a climate research assistant with access to NASA's NEX-GDDP-CMIP6 dataset.

This dataset contains 38TB of downscaled climate projections from CMIP6 models at 0.25° resolution.

Available scenarios:
- historical: 1950-2014
- ssp126: Low emissions (SSP1-2.6)
- ssp245: Middle of the road (SSP2-4.5)
- ssp370: Regional rivalry (SSP3-7.0)
- ssp585: Fossil-fueled development (SSP5-8.5)

When users ask about climate trends:
1. Clarify the region and time period
2. Load data using load_climate_data
3. Convert units if needed
4. Run analysis (trends, heatwaves, etc.)
5. Generate visualizations

Always explain results in accessible language for non-scientists.""",
    )

    return model


def chat_loop():
    """Interactive chat loop with Gemini and RCMES tools."""
    print("=" * 60)
    print("RCMES Climate Research Assistant (Powered by Gemini)")
    print("=" * 60)
    print("\nI can help you analyze NASA's NEX-GDDP-CMIP6 climate data.")
    print("Ask me questions like:")
    print("  - 'What climate models are available?'")
    print("  - 'What is the heatwave trend in California under SSP5-8.5?'")
    print("  - 'Compare temperature projections for Texas'")
    print("\nType 'quit' to exit.\n")

    try:
        model = create_gemini_model()
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        print("\nMake sure GEMINI_API_KEY is set in your environment.")
        return

    chat = model.start_chat(history=[])
    tools = get_tool_descriptions()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        try:
            # Send message to Gemini
            response = chat.send_message(
                user_input,
                tools=tools,
            )

            # Check for function calls
            for part in response.parts:
                if hasattr(part, "function_call") and part.function_call:
                    func_call = part.function_call
                    tool_name = func_call.name
                    args = dict(func_call.args)

                    print(f"\n[Calling {tool_name}...]")
                    result = execute_tool(tool_name, args)

                    # Send function result back to Gemini
                    response = chat.send_message(
                        f"Function {tool_name} returned: {json.dumps(result, default=str)}"
                    )

            # Print final response
            print(f"\nAssistant: {response.text}")

        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    chat_loop()


if __name__ == "__main__":
    main()
