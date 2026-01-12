"""
RCMES-MCP Streamlit Demo Application

Interactive web interface for exploring NASA's NEX-GDDP-CMIP6 climate data.
Provides both form-based controls and a chat interface.

Usage:
    streamlit run src/rcmes_mcp/demo_app.py --server.port 8502
    # Or via entry point after pip install:
    rcmes-demo
"""

from __future__ import annotations

import base64
import os
from datetime import date
from pathlib import Path

import streamlit as st

# Import RCMES tools directly
from rcmes_mcp.tools import data_access, processing, analysis, indices, visualization

# Load favicon
favicon_path = Path(__file__).parent / "static" / "favicon.svg"
favicon = favicon_path.read_text() if favicon_path.exists() else "R"

# Page configuration
st.set_page_config(
    page_title="RCMES Data Explorer",
    page_icon=favicon_path.as_posix() if favicon_path.exists() else "R",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_dataset_id" not in st.session_state:
    st.session_state.current_dataset_id = None


def get_models():
    """Get available climate models."""
    result = data_access.list_available_models()
    return result.get("models", [])


def get_variables():
    """Get available climate variables."""
    result = data_access.list_available_variables()
    return result.get("variables", [])


def get_scenarios():
    """Get available emissions scenarios."""
    result = data_access.list_available_scenarios()
    return result.get("scenarios", [])


def display_image(image_base64: str):
    """Display a base64-encoded image."""
    st.image(f"data:image/png;base64,{image_base64}")


def add_message(role: str, content: str, image: str | None = None):
    """Add a message to the chat history."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "image": image,
    })


# Header
st.title("RCMES Climate Data Explorer")
st.markdown("Explore NASA's NEX-GDDP-CMIP6 dataset - 38TB of global climate projections")

# Sidebar - Data Selection Controls
with st.sidebar:
    st.header("Data Selection")

    # Cache the dropdown options
    @st.cache_data(ttl=3600)
    def load_options():
        models = get_models()
        variables = get_variables()
        scenarios = get_scenarios()
        return models, variables, scenarios

    try:
        models, variables, scenarios = load_options()
    except Exception as e:
        st.error(f"Failed to load options: {e}")
        models, variables, scenarios = [], [], []

    # Model selection
    model_names = models if isinstance(models, list) and models and isinstance(models[0], str) else [m.get("name", m) if isinstance(m, dict) else str(m) for m in models]
    selected_model = st.selectbox(
        "Climate Model",
        options=model_names[:10] if model_names else ["ACCESS-CM2"],
        index=0,
        help="Select a CMIP6 climate model"
    )

    # Variable selection
    var_options = [(v["name"], f"{v['name']} - {v['long_name']}") for v in variables] if variables else [("tasmax", "tasmax - Maximum Temperature")]
    selected_var = st.selectbox(
        "Variable",
        options=[v[0] for v in var_options],
        format_func=lambda x: next((v[1] for v in var_options if v[0] == x), x),
        help="Select climate variable to analyze"
    )

    # Scenario selection - default to ssp585 for future projections (2050 default dates)
    scenario_options = [(s["id"], f"{s['id']} - {s['description'][:30]}...") for s in scenarios] if scenarios else [("ssp585", "ssp585 - High emissions")]
    scenario_ids = [s[0] for s in scenario_options]
    default_scenario_idx = scenario_ids.index("ssp585") if "ssp585" in scenario_ids else 0
    selected_scenario = st.selectbox(
        "Emissions Scenario",
        options=scenario_ids,
        index=default_scenario_idx,
        format_func=lambda x: next((s[1] for s in scenario_options if s[0] == x), x),
        help="Select emissions pathway (use SSP scenarios for dates after 2014)"
    )

    st.divider()
    st.subheader("Region")

    # Preset regions - California first as default working example
    presets = {
        "California": (32.0, 42.0, -124.0, -114.0),
        "Texas": (25.5, 36.5, -106.5, -93.5),
        "Florida": (24.5, 31.0, -87.5, -80.0),
        "Global": (-60.0, 90.0, -180.0, 180.0),
        "Custom": (None, None, None, None),
    }

    preset = st.selectbox("Preset Region", options=list(presets.keys()))
    preset_vals = presets[preset]

    col1, col2 = st.columns(2)
    with col1:
        lat_min = st.number_input("Lat Min", value=preset_vals[0] or 32.0, min_value=-60.0, max_value=90.0)
        lon_min = st.number_input("Lon Min", value=preset_vals[2] or -124.0, min_value=-180.0, max_value=180.0)
    with col2:
        lat_max = st.number_input("Lat Max", value=preset_vals[1] or 42.0, min_value=-60.0, max_value=90.0)
        lon_max = st.number_input("Lon Max", value=preset_vals[3] or -114.0, min_value=-180.0, max_value=180.0)

    st.divider()
    st.subheader("Time Period")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date(2050, 1, 1), min_value=date(1950, 1, 1), max_value=date(2100, 12, 31))
    with col2:
        end_date = st.date_input("End Date", value=date(2050, 12, 31), min_value=date(1950, 1, 1), max_value=date(2100, 12, 31))

    st.divider()

    # Load Data Button
    if st.button("Load Data", type="primary", use_container_width=True):
        with st.spinner("Loading climate data from AWS S3..."):
            result = data_access.load_climate_data(
                variable=selected_var,
                model=selected_model,
                scenario=selected_scenario,
                start_date=str(start_date),
                end_date=str(end_date),
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )

            if "error" in result:
                st.error(result["error"])
                add_message("assistant", f"Failed to load data: {result['error']}")
            else:
                dataset_id = result["dataset_id"]
                st.session_state.current_dataset_id = dataset_id
                dims = result.get("dimensions", {})
                msg = f"Loaded {selected_var} from {selected_model} ({selected_scenario})\n\n"
                msg += f"**Dataset ID:** `{dataset_id}`\n"
                msg += f"**Dimensions:** {dims.get('time', 0)} time steps, {dims.get('lat', 0)}x{dims.get('lon', 0)} grid"
                st.success(f"Dataset loaded: {dataset_id}")
                add_message("assistant", msg)

    st.divider()
    st.subheader("Analysis")

    analysis_type = st.selectbox(
        "Analysis Type",
        options=["Statistics", "Trend", "Climatology", "Heatwaves", "Regional Mean"],
    )

    if st.button("Run Analysis", use_container_width=True):
        if not st.session_state.current_dataset_id:
            st.warning("Load data first!")
        else:
            ds_id = st.session_state.current_dataset_id
            with st.spinner(f"Running {analysis_type}..."):
                if analysis_type == "Statistics":
                    result = analysis.calculate_statistics(dataset_id=ds_id)
                elif analysis_type == "Trend":
                    result = analysis.calculate_trend(dataset_id=ds_id)
                elif analysis_type == "Climatology":
                    result = analysis.calculate_climatology(dataset_id=ds_id)
                    if "dataset_id" in result:
                        st.session_state.current_dataset_id = result["dataset_id"]
                elif analysis_type == "Heatwaves":
                    result = indices.analyze_heatwaves(dataset_id=ds_id)
                    if "dataset_id" in result:
                        st.session_state.current_dataset_id = result["dataset_id"]
                elif analysis_type == "Regional Mean":
                    result = analysis.calculate_regional_mean(dataset_id=ds_id)
                    if "dataset_id" in result:
                        st.session_state.current_dataset_id = result["dataset_id"]

                if "error" in result:
                    st.error(result["error"])
                    add_message("assistant", f"Analysis failed: {result['error']}")
                else:
                    # Format result for display
                    if analysis_type == "Statistics":
                        var = result.get('variable', 'Data')
                        msg = f"**Statistics for {var}:**\n\n"
                        msg += f"| Metric | Value |\n|--------|-------|\n"
                        msg += f"| Mean | {result.get('mean', 0):.2f} |\n"
                        msg += f"| Std Dev | {result.get('std', 0):.2f} |\n"
                        msg += f"| Min | {result.get('min', 0):.2f} |\n"
                        msg += f"| Max | {result.get('max', 0):.2f} |\n"
                        if 'percentiles' in result:
                            p = result['percentiles']
                            msg += f"| 5th Percentile | {p.get('p5', 0):.2f} |\n"
                            msg += f"| 25th Percentile | {p.get('p25', 0):.2f} |\n"
                            msg += f"| Median (50th) | {p.get('p50', 0):.2f} |\n"
                            msg += f"| 75th Percentile | {p.get('p75', 0):.2f} |\n"
                            msg += f"| 95th Percentile | {p.get('p95', 0):.2f} |\n"
                    elif analysis_type == "Trend":
                        var = result.get('variable', 'Data')
                        msg = f"**Trend Analysis for {var}:**\n\n"
                        msg += f"| Metric | Value |\n|--------|-------|\n"
                        trend = result.get('trend', result)  # Handle nested or flat structure
                        if 'slope_per_decade' in trend:
                            unit = trend.get('unit', 'per decade')
                            msg += f"| Trend | {trend['slope_per_decade']:.4f} {unit} |\n"
                        if 'p_value' in trend:
                            p_val = trend['p_value']
                            sig = "Yes" if p_val < 0.05 else "No"
                            msg += f"| P-value | {p_val:.2e} |\n"
                            msg += f"| Significant (p<0.05) | {sig} |\n"
                        if 'r_squared' in trend:
                            msg += f"| R-squared | {trend['r_squared']:.4f} |\n"
                        if 'interpretation' in result:
                            msg += f"\n{result['interpretation']}"
                    else:
                        msg = f"**{analysis_type} Results:**\n\nCompleted successfully. New dataset ID: `{result.get('dataset_id', 'N/A')}`"
                    add_message("assistant", msg)

    st.divider()
    st.subheader("Visualization")

    viz_type = st.selectbox(
        "Plot Type",
        options=["Map", "Time Series", "Histogram"],
    )

    if st.button("Generate Plot", use_container_width=True):
        if not st.session_state.current_dataset_id:
            st.warning("Load data first!")
        else:
            ds_id = st.session_state.current_dataset_id
            with st.spinner(f"Generating {viz_type}..."):
                if viz_type == "Map":
                    result = visualization.generate_map(
                        dataset_id=ds_id,
                        title=f"{selected_var} - {selected_model} ({selected_scenario})"
                    )
                elif viz_type == "Time Series":
                    result = visualization.generate_timeseries_plot(
                        dataset_ids=[ds_id],
                        title=f"{selected_var} Time Series",
                        show_trend=True
                    )
                elif viz_type == "Histogram":
                    result = visualization.generate_histogram(
                        dataset_id=ds_id,
                        title=f"{selected_var} Distribution"
                    )

                if "error" in result:
                    st.error(result["error"])
                    add_message("assistant", f"Visualization failed: {result['error']}")
                elif "image_base64" in result:
                    add_message("assistant", f"Generated {viz_type}:", image=result["image_base64"])

# Main content area
col_main, col_info = st.columns([3, 1])

with col_main:
    st.subheader("Conversation")

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("image"):
                display_image(msg["image"])

    # Show example queries when data is loaded
    if st.session_state.current_dataset_id:
        st.markdown("**Try these queries:**")
        example_cols = st.columns(4)
        examples = [
            "Show statistics",
            "Calculate trend",
            "Generate map",
            "Plot time series"
        ]
        for i, example in enumerate(examples):
            if example_cols[i].button(example, key=f"example_{i}"):
                add_message("user", example)
                ds_id = st.session_state.current_dataset_id

                if "statistics" in example.lower():
                    result = analysis.calculate_statistics(dataset_id=ds_id)
                    if "error" not in result:
                        var = result.get('variable', 'Data')
                        msg = f"**Statistics for {var}:**\n\n"
                        msg += f"| Metric | Value |\n|--------|-------|\n"
                        msg += f"| Mean | {result.get('mean', 0):.2f} |\n"
                        msg += f"| Std Dev | {result.get('std', 0):.2f} |\n"
                        msg += f"| Min | {result.get('min', 0):.2f} |\n"
                        msg += f"| Max | {result.get('max', 0):.2f} |\n"
                        if 'percentiles' in result:
                            p = result['percentiles']
                            msg += f"| 5th Percentile | {p.get('p5', 0):.2f} |\n"
                            msg += f"| 25th Percentile | {p.get('p25', 0):.2f} |\n"
                            msg += f"| Median (50th) | {p.get('p50', 0):.2f} |\n"
                            msg += f"| 75th Percentile | {p.get('p75', 0):.2f} |\n"
                            msg += f"| 95th Percentile | {p.get('p95', 0):.2f} |\n"
                        add_message("assistant", msg)
                    else:
                        add_message("assistant", f"Error: {result['error']}")

                elif "trend" in example.lower():
                    result = analysis.calculate_trend(dataset_id=ds_id)
                    if "error" not in result:
                        var = result.get('variable', 'Data')
                        msg = f"**Trend Analysis for {var}:**\n\n"
                        msg += f"| Metric | Value |\n|--------|-------|\n"
                        trend = result.get('trend', result)
                        if 'slope_per_decade' in trend:
                            unit = trend.get('unit', 'per decade')
                            msg += f"| Trend | {trend['slope_per_decade']:.4f} {unit} |\n"
                        if 'p_value' in trend:
                            p_val = trend['p_value']
                            sig = "Yes" if p_val < 0.05 else "No"
                            msg += f"| P-value | {p_val:.2e} |\n"
                            msg += f"| Significant (p<0.05) | {sig} |\n"
                        if 'r_squared' in trend:
                            msg += f"| R-squared | {trend['r_squared']:.4f} |\n"
                        if 'interpretation' in result:
                            msg += f"\n{result['interpretation']}"
                        add_message("assistant", msg)
                    else:
                        add_message("assistant", f"Error: {result['error']}")

                elif "map" in example.lower():
                    result = visualization.generate_map(dataset_id=ds_id)
                    if "error" not in result:
                        add_message("assistant", "Generated map:", image=result.get("image_base64"))
                    else:
                        add_message("assistant", f"Error: {result['error']}")

                elif "time series" in example.lower():
                    result = visualization.generate_timeseries_plot(dataset_ids=[ds_id], show_trend=True)
                    if "error" not in result:
                        add_message("assistant", "Generated time series:", image=result.get("image_base64"))
                    else:
                        add_message("assistant", f"Error: {result['error']}")

                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about climate data..."):
        add_message("user", prompt)

        # Simple keyword-based responses (no LLM required for demo)
        response = None
        prompt_lower = prompt.lower()

        if "model" in prompt_lower and ("available" in prompt_lower or "list" in prompt_lower):
            result = data_access.list_available_models()
            response = f"Available models: {', '.join(result.get('models', [])[:10])}..."
        elif "variable" in prompt_lower and ("available" in prompt_lower or "list" in prompt_lower):
            result = data_access.list_available_variables()
            vars_list = [f"{v['name']} ({v['long_name']})" for v in result.get("variables", [])[:5]]
            response = f"Available variables:\n- " + "\n- ".join(vars_list)
        elif "scenario" in prompt_lower:
            result = data_access.list_available_scenarios()
            scen_list = [f"{s['id']}: {s['description']}" for s in result.get("scenarios", [])]
            response = f"Available scenarios:\n- " + "\n- ".join(scen_list)
        elif "loaded" in prompt_lower or "dataset" in prompt_lower:
            result = data_access.list_loaded_datasets()
            if result.get("dataset_count", 0) > 0:
                datasets = result.get("datasets", [])
                ds_list = [f"`{d['id']}`: {d.get('variable', '')} from {d.get('model', '')} ({d.get('scenario', '')})" for d in datasets]
                response = f"Loaded datasets:\n- " + "\n- ".join(ds_list)
            else:
                response = "No datasets loaded. Use the sidebar to load data."
        elif "help" in prompt_lower:
            response = """**How to use this demo:**

1. **Load Data**: Use the sidebar to select a model, variable, scenario, region, and time period. Click "Load Data".

2. **Analyze**: After loading data, choose an analysis type and click "Run Analysis".

3. **Visualize**: Generate maps, time series, or histograms of your data.

4. **Chat**: Ask me about available models, variables, scenarios, or loaded datasets.

**Example queries:**
- "What models are available?"
- "List the variables"
- "Show me the scenarios"
- "What datasets are loaded?"
- "What is the heatwave trend in California?"
"""
        # Handle natural language climate questions
        elif any(word in prompt_lower for word in ["heatwave", "heat wave", "temperature", "trend", "climate", "warming"]):
            # Detect region
            regions = {
                "california": (32.0, 42.0, -124.0, -114.0),
                "texas": (25.5, 36.5, -106.5, -93.5),
                "florida": (24.5, 31.0, -87.5, -80.0),
            }
            region_name = "California"
            region_bounds = regions["california"]
            for name, bounds in regions.items():
                if name in prompt_lower:
                    region_name = name.title()
                    region_bounds = bounds
                    break

            # Determine variable based on query
            variable = "tasmax"  # Default to max temperature for heatwave queries
            if "precipitation" in prompt_lower or "rain" in prompt_lower:
                variable = "pr"
            elif "humidity" in prompt_lower:
                variable = "hurs"

            with st.spinner(f"Loading data and analyzing for {region_name}..."):
                # Load data
                load_result = data_access.load_climate_data(
                    variable=variable,
                    model="ACCESS-CM2",
                    scenario="ssp585",
                    start_date="2050-01-01",
                    end_date="2050-12-31",
                    lat_min=region_bounds[0],
                    lat_max=region_bounds[1],
                    lon_min=region_bounds[2],
                    lon_max=region_bounds[3],
                )

                if "error" in load_result:
                    response = f"Error loading data: {load_result['error']}"
                else:
                    ds_id = load_result["dataset_id"]
                    st.session_state.current_dataset_id = ds_id

                    # Run appropriate analysis
                    if "heatwave" in prompt_lower or "heat wave" in prompt_lower:
                        result = indices.analyze_heatwaves(dataset_id=ds_id)
                        if "error" not in result:
                            response = f"**Heatwave Analysis for {region_name} (2050, SSP5-8.5):**\n\n"
                            response += f"Model: ACCESS-CM2 | Variable: Daily Max Temperature\n\n"
                            if "summary" in result:
                                s = result["summary"]
                                response += f"| Metric | Value |\n|--------|-------|\n"
                                if s.get('mean_annual_hot_days') is not None:
                                    response += f"| Hot Days (>90th percentile) | {s['mean_annual_hot_days']:.1f} days/year |\n"
                                if s.get('mean_annual_heatwave_frequency') is not None:
                                    response += f"| Heatwave Events | {s['mean_annual_heatwave_frequency']:.1f} per year |\n"
                                if s.get('mean_heatwave_duration') is not None:
                                    response += f"| Avg Duration | {s['mean_heatwave_duration']:.1f} days |\n"
                            response += f"\nDataset ID: `{result.get('dataset_id', ds_id)}`"
                        else:
                            response = f"Error analyzing heatwaves: {result['error']}"
                    else:
                        # General trend analysis
                        result = analysis.calculate_trend(dataset_id=ds_id)
                        if "error" not in result:
                            response = f"**Trend Analysis for {region_name} (2050, SSP5-8.5):**\n\n"
                            response += f"| Metric | Value |\n|--------|-------|\n"
                            trend = result.get('trend', result)
                            if 'slope_per_decade' in trend:
                                unit = trend.get('unit', 'per decade')
                                response += f"| Trend | {trend['slope_per_decade']:.4f} {unit} |\n"
                            if 'p_value' in trend:
                                p_val = trend['p_value']
                                sig = "Yes" if p_val < 0.05 else "No"
                                response += f"| P-value | {p_val:.2e} |\n"
                                response += f"| Significant | {sig} |\n"
                            if 'interpretation' in result:
                                response += f"\n{result['interpretation']}"
                        else:
                            response = f"Error calculating trend: {result['error']}"
        else:
            response = "I can help with climate data analysis. Try asking:\n- 'What is the heatwave trend in California?'\n- 'What models are available?'\n- 'Show statistics' (after loading data)\n\nOr use the sidebar to load and analyze data."

        add_message("assistant", response)
        st.rerun()

with col_info:
    st.subheader("Loaded Datasets")

    result = data_access.list_loaded_datasets()
    datasets = result.get("datasets", [])

    if datasets:
        for ds in datasets:
            with st.expander(f"`{ds['id']}`", expanded=False):
                st.write(f"**Variable:** {ds.get('variable', 'N/A')}")
                st.write(f"**Model:** {ds.get('model', 'N/A')}")
                st.write(f"**Scenario:** {ds.get('scenario', 'N/A')}")
                if ds.get("time_range"):
                    st.write(f"**Time:** {ds['time_range'][0]} to {ds['time_range'][1]}")
                if st.button("Select", key=f"select_{ds['id']}"):
                    st.session_state.current_dataset_id = ds['id']
                    st.rerun()
    else:
        st.info("No datasets loaded yet. Use the sidebar to load data.")

    if st.session_state.current_dataset_id:
        st.divider()
        st.write(f"**Current:** `{st.session_state.current_dataset_id}`")


def main():
    """Entry point for the demo app."""
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        __file__,
        "--server.port", "8502",
    ])


if __name__ == "__main__":
    # This block runs when executed directly with streamlit run
    pass
