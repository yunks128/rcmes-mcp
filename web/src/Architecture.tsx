import { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

// Initialize mermaid
mermaid.initialize({
  startOnLoad: false,
  theme: 'neutral',
  securityLevel: 'loose',
});

function MermaidDiagram({ chart, id }: { chart: string; id: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.innerHTML = '';
      mermaid.render(id, chart).then(({ svg }) => {
        if (containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      });
    }
  }, [chart, id]);

  return <div ref={containerRef} className="mermaid-container" />;
}

export default function Architecture() {
  return (
    <div className="architecture-page">
      <h1>Architecture & Code-to-Data</h1>

      <section className="arch-section">
        <h2>The Problem: Data is Too Big to Move</h2>
        <p>Traditional climate analysis requires downloading data to your local machine:</p>

        <div className="comparison-grid">
          <div className="comparison-card error">
            <h3>Traditional Approach</h3>
            <ol>
              <li>Download 38TB dataset to local disk</li>
              <li>Wait days/weeks for transfer</li>
              <li>Need expensive storage</li>
              <li>Process locally</li>
              <li>Results limited by your hardware</li>
            </ol>
          </div>
          <div className="comparison-card success">
            <h3>Code-to-Data Approach</h3>
            <ol>
              <li>Data stays in the cloud (AWS S3)</li>
              <li>Send only your query</li>
              <li>Cloud processes the subset</li>
              <li>Download only results (KB/MB)</li>
              <li>Scales with cloud resources</li>
            </ol>
          </div>
        </div>
      </section>

      <section className="arch-section">
        <h2>How RCMES-MCP Works</h2>
        <MermaidDiagram
          id="flow-diagram"
          chart={`
flowchart TB
    subgraph Browser["Browser - React UI"]
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
          `}
        />
      </section>

      <section className="arch-section">
        <h2>Lazy Loading in Action</h2>
        <p>When you click "Load Data", here's what happens:</p>

        <pre className="code-block">
{`# Step 1: Open dataset (NO DATA DOWNLOADED YET)
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
# Only now does it fetch the ~6MB subset from S3`}
        </pre>

        <div className="metrics-row">
          <div className="metric-card">
            <div className="metric-value">38 TB</div>
            <div className="metric-label">Full Dataset</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">~6 MB</div>
            <div className="metric-label">California 2050</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">99.99998%</div>
            <div className="metric-label">Reduction</div>
          </div>
        </div>
      </section>

      <section className="arch-section">
        <h2>The MCP Protocol</h2>

        <h3>What is MCP? (Simple Explanation)</h3>
        <p>
          <strong>MCP (Model Context Protocol)</strong> is like giving an AI assistant a toolbox.
        </p>

        <table className="info-table">
          <thead>
            <tr>
              <th>Without MCP</th>
              <th>With MCP</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>AI can only talk</td>
              <td>AI can talk AND do things</td>
            </tr>
            <tr>
              <td>"I don't have access to that data"</td>
              <td>"Let me fetch that data for you"</td>
            </tr>
            <tr>
              <td>Limited to what it was trained on</td>
              <td>Can use live tools and real data</td>
            </tr>
          </tbody>
        </table>

        <h3>How it works:</h3>
        <ol>
          <li><strong>You ask a question</strong> in plain English</li>
          <li><strong>AI figures out</strong> what tools it needs</li>
          <li><strong>AI calls the tools</strong> (like <code>load_climate_data</code>, <code>analyze_heatwaves</code>)</li>
          <li><strong>Tools return results</strong> to the AI</li>
          <li><strong>AI explains the results</strong> back to you</li>
        </ol>

        <h3>MCP in Action</h3>
        <MermaidDiagram
          id="mcp-sequence"
          chart={`
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
          `}
        />
      </section>

      <section className="arch-section">
        <h2>Available Tools</h2>
        <table className="info-table">
          <thead>
            <tr>
              <th>Category</th>
              <th>Tools</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Data Access</strong></td>
              <td><code>load_climate_data</code>, <code>list_available_models</code>, <code>list_available_variables</code></td>
            </tr>
            <tr>
              <td><strong>Processing</strong></td>
              <td><code>temporal_subset</code>, <code>spatial_subset</code>, <code>regrid</code>, <code>convert_units</code></td>
            </tr>
            <tr>
              <td><strong>Analysis</strong></td>
              <td><code>calculate_statistics</code>, <code>calculate_trend</code>, <code>calculate_climatology</code></td>
            </tr>
            <tr>
              <td><strong>Indices</strong></td>
              <td><code>analyze_heatwaves</code>, <code>calculate_drought_index</code>, <code>calculate_etccdi_index</code></td>
            </tr>
            <tr>
              <td><strong>Visualization</strong></td>
              <td><code>generate_map</code>, <code>generate_timeseries_plot</code>, <code>generate_taylor_diagram</code></td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="arch-section">
        <h2>Dataset: NEX-GDDP-CMIP6</h2>

        <div className="comparison-grid">
          <div className="info-card">
            <h3>What is it?</h3>
            <p>
              NASA's <strong>NEX-GDDP-CMIP6</strong> (NASA Earth Exchange Global Daily Downscaled Projections)
              provides high-resolution climate projections derived from CMIP6 models.
            </p>
            <ul>
              <li><strong>Downscaled</strong> from ~100km to ~25km resolution</li>
              <li><strong>Bias-corrected</strong> against observations</li>
              <li><strong>Daily</strong> temporal resolution</li>
              <li><strong>Global</strong> coverage (land areas)</li>
            </ul>
          </div>
          <div className="info-card">
            <h3>Specifications</h3>
            <table className="specs-table">
              <tbody>
                <tr><td>Size</td><td>38 TB</td></tr>
                <tr><td>Resolution</td><td>0.25 deg (~25 km)</td></tr>
                <tr><td>Time Period</td><td>1950-2100</td></tr>
                <tr><td>Models</td><td>35 CMIP6 GCMs</td></tr>
                <tr><td>Scenarios</td><td>Historical + 4 SSPs</td></tr>
                <tr><td>Variables</td><td>9 (temp, precip, humidity, etc.)</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <h3>Emissions Scenarios (SSPs)</h3>
        <table className="info-table">
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Name</th>
              <th>Description</th>
              <th>2100 Warming</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>SSP1-2.6</strong></td>
              <td>Sustainability</td>
              <td>Low emissions, green growth</td>
              <td>~1.8°C</td>
            </tr>
            <tr>
              <td><strong>SSP2-4.5</strong></td>
              <td>Middle of the Road</td>
              <td>Moderate emissions</td>
              <td>~2.7°C</td>
            </tr>
            <tr>
              <td><strong>SSP3-7.0</strong></td>
              <td>Regional Rivalry</td>
              <td>High emissions, fragmentation</td>
              <td>~3.6°C</td>
            </tr>
            <tr>
              <td><strong>SSP5-8.5</strong></td>
              <td>Fossil-fueled Development</td>
              <td>Very high emissions</td>
              <td>~4.4°C</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="arch-section">
        <h2>System Architecture</h2>
        <p>The RCMES system uses a modern React + FastAPI architecture:</p>
        <MermaidDiagram
          id="system-arch"
          chart={`
flowchart TB
    subgraph Client["Client Layer"]
        Browser["Browser"]
        React["React UI<br/>(Vite + TypeScript)"]
    end

    subgraph API["API Layer"]
        FastAPI["FastAPI Server<br/>:8502"]
        REST["REST Endpoints<br/>/api/*"]
    end

    subgraph Core["Core Layer"]
        subgraph Tools["RCMES Tools"]
            DA["data_access.py"]
            PR["processing.py"]
            AN["analysis.py"]
            IX["indices.py"]
            VZ["visualization.py"]
        end
        Session["Session Manager<br/>(In-Memory Cache)"]
    end

    subgraph Compute["Compute Layer"]
        XArray["xarray"]
        Dask["Dask<br/>(Lazy Loading)"]
        XClim["xclim<br/>(Climate Indices)"]
    end

    subgraph Data["Data Layer"]
        S3FS["s3fs / fsspec"]
        S3[("AWS S3<br/>NEX-GDDP-CMIP6<br/>38 TB")]
    end

    Browser --> React
    React -->|HTTP/JSON| FastAPI
    FastAPI --> REST
    REST --> Tools
    Tools --> Session
    Session --> XArray
    XArray --> Dask
    Dask --> XClim
    XArray --> S3FS
    S3FS -->|"Lazy fetch<br/>only needed chunks"| S3
          `}
        />

        <h3>Request Flow</h3>
        <MermaidDiagram
          id="request-flow"
          chart={`
sequenceDiagram
    participant Browser
    participant React
    participant FastAPI
    participant Tools
    participant S3

    Browser->>React: Click "Load Data"
    React->>FastAPI: POST /api/load-data
    FastAPI->>Tools: load_climate_data()
    Tools->>S3: Open dataset (metadata only)
    S3-->>Tools: File handles
    Tools->>Tools: Apply spatial/temporal subset
    Tools-->>FastAPI: dataset_id
    FastAPI-->>React: JSON response
    React->>FastAPI: POST /api/visualize
    FastAPI->>Tools: generate_map()
    Tools->>S3: Fetch actual data chunks
    S3-->>Tools: ~6MB subset
    Tools->>Tools: Generate plot
    Tools-->>FastAPI: base64 image
    FastAPI-->>React: JSON with image
    React-->>Browser: Display results
          `}
        />
      </section>

      <section className="arch-section">
        <h2>Why This Matters</h2>

        <h3>Democratizing Climate Data</h3>
        <table className="info-table">
          <thead>
            <tr>
              <th>Before RCMES-MCP</th>
              <th>With RCMES-MCP</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Required climate science expertise</td>
              <td>Ask in natural language</td>
            </tr>
            <tr>
              <td>Needed to write Python/NCO scripts</td>
              <td>No coding required</td>
            </tr>
            <tr>
              <td>Had to download terabytes of data</td>
              <td>Data stays in the cloud</td>
            </tr>
            <tr>
              <td>Limited to those with compute resources</td>
              <td>Instant results on-demand</td>
            </tr>
          </tbody>
        </table>

        <h3>Use Cases</h3>
        <ul>
          <li><strong>Policymakers:</strong> Quick regional climate projections for planning</li>
          <li><strong>Researchers:</strong> Rapid exploratory analysis before deep dives</li>
          <li><strong>Educators:</strong> Interactive climate data for teaching</li>
          <li><strong>Journalists:</strong> Data-driven climate reporting</li>
          <li><strong>Developers:</strong> Build climate-aware applications</li>
        </ul>
      </section>

      <section className="arch-section">
        <h2>Technical Stack</h2>

        <div className="comparison-grid">
          <div className="info-card">
            <h3>Frontend</h3>
            <table className="specs-table">
              <tbody>
                <tr><td>Framework</td><td>React 18</td></tr>
                <tr><td>Build Tool</td><td>Vite</td></tr>
                <tr><td>Language</td><td>TypeScript</td></tr>
                <tr><td>Diagrams</td><td>Mermaid.js</td></tr>
                <tr><td>Styling</td><td>CSS (custom)</td></tr>
              </tbody>
            </table>
          </div>
          <div className="info-card">
            <h3>Backend</h3>
            <table className="specs-table">
              <tbody>
                <tr><td>API Framework</td><td>FastAPI</td></tr>
                <tr><td>ASGI Server</td><td>Uvicorn</td></tr>
                <tr><td>MCP Server</td><td>FastMCP</td></tr>
                <tr><td>Language</td><td>Python 3.10+</td></tr>
                <tr><td>Validation</td><td>Pydantic</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="comparison-grid">
          <div className="info-card">
            <h3>Data Processing</h3>
            <table className="specs-table">
              <tbody>
                <tr><td>Array Library</td><td>xarray</td></tr>
                <tr><td>Parallel Computing</td><td>Dask</td></tr>
                <tr><td>Climate Indices</td><td>xclim</td></tr>
                <tr><td>Statistics</td><td>SciPy</td></tr>
                <tr><td>Geospatial</td><td>GeoPandas, Shapely</td></tr>
              </tbody>
            </table>
          </div>
          <div className="info-card">
            <h3>Data Access</h3>
            <table className="specs-table">
              <tbody>
                <tr><td>Cloud Storage</td><td>AWS S3</td></tr>
                <tr><td>File System</td><td>fsspec, s3fs</td></tr>
                <tr><td>File Formats</td><td>NetCDF4, HDF5, Zarr</td></tr>
                <tr><td>NetCDF Engine</td><td>h5netcdf</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="comparison-grid">
          <div className="info-card">
            <h3>Visualization</h3>
            <table className="specs-table">
              <tbody>
                <tr><td>Plotting</td><td>Matplotlib</td></tr>
                <tr><td>Maps</td><td>Cartopy</td></tr>
                <tr><td>Output</td><td>PNG (base64)</td></tr>
              </tbody>
            </table>
          </div>
          <div className="info-card">
            <h3>Deployment</h3>
            <table className="specs-table">
              <tbody>
                <tr><td>Process Manager</td><td>systemd</td></tr>
                <tr><td>Port</td><td>8502</td></tr>
                <tr><td>Static Files</td><td>FastAPI StaticFiles</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <h3>Technology Flow</h3>
        <MermaidDiagram
          id="tech-stack"
          chart={`
flowchart LR
    subgraph Frontend
        React[React + TypeScript]
        Vite[Vite Build]
        Mermaid[Mermaid.js]
    end

    subgraph Backend
        FastAPI[FastAPI]
        Uvicorn[Uvicorn ASGI]
        Pydantic[Pydantic]
    end

    subgraph Processing
        xarray[xarray]
        Dask[Dask]
        xclim[xclim]
        SciPy[SciPy]
    end

    subgraph Visualization
        Matplotlib[Matplotlib]
        Cartopy[Cartopy]
    end

    subgraph Storage
        s3fs[s3fs]
        h5netcdf[h5netcdf]
        S3[(AWS S3)]
    end

    React --> FastAPI
    FastAPI --> xarray
    xarray --> Dask
    Dask --> xclim
    xarray --> s3fs
    s3fs --> h5netcdf
    h5netcdf --> S3
    xarray --> Matplotlib
    Matplotlib --> Cartopy
          `}
        />
      </section>

      <footer className="arch-footer">
        <p>
          <strong>References:</strong>{' '}
          <a href="https://registry.opendata.aws/nex-gddp-cmip6/" target="_blank" rel="noopener noreferrer">
            NEX-GDDP-CMIP6 on AWS
          </a>{' '}
          |{' '}
          <a href="https://modelcontextprotocol.io/" target="_blank" rel="noopener noreferrer">
            Model Context Protocol
          </a>{' '}
          |{' '}
          <a href="https://climate.apache.org/" target="_blank" rel="noopener noreferrer">
            Apache Open Climate Workbench
          </a>
        </p>
      </footer>
    </div>
  );
}
