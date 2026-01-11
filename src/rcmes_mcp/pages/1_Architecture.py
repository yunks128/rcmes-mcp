"""
Architecture & Code-to-Data Concept

This page explains how RCMES-MCP moves computation to the data
rather than moving data to computation.
"""

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Architecture - RCMES",
    page_icon="A",
    layout="wide",
)


def mermaid(code: str, height: int = 400) -> None:
    """Render a Mermaid diagram using mermaid.js CDN."""
    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true, theme:'neutral'}});</script>
    <div class="mermaid">
    {code}
    </div>
    """
    components.html(html, height=height)


st.title("Architecture & Code-to-Data")

st.markdown("""
## The Problem: Data is Too Big to Move

Traditional climate analysis requires downloading data to your local machine:
""")

col1, col2 = st.columns(2)

with col1:
    st.error("""
    **Traditional Approach**

    1. Download 38TB dataset to local disk
    2. Wait days/weeks for transfer
    3. Need expensive storage
    4. Process locally
    5. Results limited by your hardware
    """)

with col2:
    st.success("""
    **Code-to-Data Approach**

    1. Data stays in the cloud (AWS S3)
    2. Send only your query
    3. Cloud processes the subset
    4. Download only results (KB/MB)
    5. Scales with cloud resources
    """)

st.divider()

st.header("How RCMES-MCP Works")

mermaid("""
flowchart TB
    subgraph Browser["Browser - Streamlit UI"]
        UI[User Interface]
        Select[/"Select: Region, Time, Variable"/]
    end

    subgraph Server["RCMES-MCP Server"]
        Tools[MCP Tools]
        subgraph ToolTypes[" "]
            DA[Data Access]
            AN[Analysis]
            VZ[Visualization]
        end
        XArray["xarray + Dask (Lazy Loading)"]
    end

    subgraph S3["AWS S3: nex-gddp-cmip6"]
        Data[("38 TB Climate Data")]
    end

    UI --> Select
    Select -->|Query parameters| Tools
    Tools --> DA & AN & VZ
    DA & AN & VZ --> XArray
    XArray -->|Fetch only subset| Data
    Data -->|Subset data| XArray
    XArray -->|Results| Tools
    Tools -->|JSON + Images| UI
""", height=600)

st.divider()

st.header("Lazy Loading in Action")

st.markdown("""
When you click "Load Data", here's what happens:
""")

code_example = '''
# Step 1: Open dataset (NO DATA DOWNLOADED YET)
ds = xr.open_mfdataset(
    "s3://nex-gddp-cmip6/.../tasmax_2050.nc",
    engine="h5netcdf",
    chunks={"time": 365, "lat": 100, "lon": 100}  # Lazy chunks
)
# This returns instantly - just metadata!

# Step 2: Subset by region and time (STILL NO DATA)
ds = ds.sel(
    time=slice("2050-01-01", "2050-12-31"),
    lat=slice(32, 42),      # California latitude
    lon=slice(236, 246)     # California longitude (0-360)
)
# This just modifies the query - still no download!

# Step 3: Compute statistics (NOW DATA IS FETCHED)
mean_temp = ds.tasmax.mean().compute()
# Only now does it fetch the ~6MB subset from S3
'''

st.code(code_example, language="python")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Full Dataset", "38 TB", help="Total size of NEX-GDDP-CMIP6")

with col2:
    st.metric("California 2050", "~6 MB", help="Data actually downloaded for one query")

with col3:
    st.metric("Reduction", "99.99998%", help="Data transfer reduction")

st.divider()

st.header("The MCP Protocol")

mermaid("""
sequenceDiagram
    participant User
    participant LLM as Gemini / LLM
    participant MCP as RCMES-MCP Server
    participant S3 as AWS S3

    User->>LLM: What is the heatwave trend in California?
    LLM->>LLM: Parse intent, select tools
    LLM->>MCP: load_climate_data(tasmax, ACCESS-CM2, ssp585, ...)
    MCP->>S3: Fetch subset (lazy query)
    S3-->>MCP: Return ~6MB data
    MCP-->>LLM: dataset_id: ds_abc123
    LLM->>MCP: analyze_heatwaves(ds_abc123)
    MCP-->>LLM: trend: +8.5 days/decade
    LLM->>MCP: generate_timeseries_plot(ds_abc123)
    MCP-->>LLM: image_base64
    LLM->>User: California heatwaves increasing... + plot
""", height=550)

st.markdown("""
### Available Tools

| Category | Tools |
|----------|-------|
| **Data Access** | `load_climate_data`, `list_available_models`, `list_available_variables` |
| **Processing** | `temporal_subset`, `spatial_subset`, `regrid`, `convert_units` |
| **Analysis** | `calculate_statistics`, `calculate_trend`, `calculate_climatology` |
| **Indices** | `analyze_heatwaves`, `calculate_drought_index`, `calculate_etccdi_index` |
| **Visualization** | `generate_map`, `generate_timeseries_plot`, `generate_taylor_diagram` |
""")

st.divider()

st.header("Dataset: NEX-GDDP-CMIP6")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### What is it?

    NASA's **NEX-GDDP-CMIP6** (NASA Earth Exchange Global Daily Downscaled Projections)
    provides high-resolution climate projections derived from CMIP6 models.

    - **Downscaled** from ~100km to ~25km resolution
    - **Bias-corrected** against observations
    - **Daily** temporal resolution
    - **Global** coverage (land areas)
    """)

with col2:
    st.markdown("""
    ### Specifications

    | Property | Value |
    |----------|-------|
    | Size | 38 TB |
    | Resolution | 0.25 deg (~25 km) |
    | Time Period | 1950-2100 |
    | Models | 35 CMIP6 GCMs |
    | Scenarios | Historical + 4 SSPs |
    | Variables | 9 (temperature, precip, humidity, etc.) |
    """)

st.markdown("""
### Emissions Scenarios (SSPs)

| Scenario | Name | Description | 2100 Warming |
|----------|------|-------------|--------------|
| **SSP1-2.6** | Sustainability | Low emissions, green growth | ~1.8C |
| **SSP2-4.5** | Middle of the Road | Moderate emissions | ~2.7C |
| **SSP3-7.0** | Regional Rivalry | High emissions, fragmentation | ~3.6C |
| **SSP5-8.5** | Fossil-fueled Development | Very high emissions | ~4.4C |
""")

st.divider()

st.header("System Architecture")

mermaid("""
flowchart LR
    subgraph Clients["Clients"]
        Gemini[Gemini]
        Claude[Claude]
        Web["Web UI (Streamlit)"]
        API[Custom Apps]
    end

    subgraph RCMES["RCMES-MCP Server"]
        FastMCP[FastMCP]
        subgraph Tools["Tool Modules"]
            T1[data_access]
            T2[processing]
            T3[analysis]
            T4[indices]
            T5[visualization]
        end
        Session["Session Manager"]
    end

    subgraph Data["Data Sources"]
        S3[("AWS S3 NEX-GDDP-CMIP6")]
        RCMED[("RCMED Observations")]
    end

    Gemini --> FastMCP
    Claude --> FastMCP
    Web --> Tools
    API --> FastMCP
    FastMCP --> Tools
    Tools --> Session
    Session --> S3
    Session -.-> RCMED
""", height=500)

st.divider()

st.header("Why This Matters")

st.markdown("""
### Democratizing Climate Data

| Before RCMES-MCP | With RCMES-MCP |
|------------------|----------------|
| Required climate science expertise | Ask in natural language |
| Needed to write Python/NCO scripts | No coding required |
| Had to download terabytes of data | Data stays in the cloud |
| Limited to those with compute resources | Instant results on-demand |

### Use Cases

- **Policymakers**: Quick regional climate projections for planning
- **Researchers**: Rapid exploratory analysis before deep dives
- **Educators**: Interactive climate data for teaching
- **Journalists**: Data-driven climate reporting
- **Developers**: Build climate-aware applications
""")

st.divider()

st.header("Technical Stack")

mermaid("""
flowchart TB
    subgraph Interface["Interface Layer"]
        MCP[FastMCP Server]
        ST[Streamlit UI]
        MPL[Matplotlib + Cartopy]
    end

    subgraph Compute["Compute Layer"]
        XA[xarray]
        DASK[Dask]
        XC[xclim]
        SP[SciPy]
    end

    subgraph Storage["Data Layer"]
        S3[AWS S3]
        ZARR[Zarr / NetCDF]
        FS[fsspec / s3fs]
    end

    MCP --> XA
    ST --> XA
    XA --> DASK
    DASK --> XC
    DASK --> SP
    XA --> FS
    FS --> ZARR
    ZARR --> S3
    XA --> MPL
""", height=550)

st.divider()

st.caption("""
**References:**
[NEX-GDDP-CMIP6 on AWS](https://registry.opendata.aws/nex-gddp-cmip6/) |
[Model Context Protocol](https://modelcontextprotocol.io/) |
[Apache Open Climate Workbench](https://climate.apache.org/)
""")
