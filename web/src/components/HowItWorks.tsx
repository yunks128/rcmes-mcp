export default function HowItWorks({ onClose }: { onClose: () => void }) {
  return (
    <div className="hiw-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label="How It Works">
      <div className="hiw-panel" onClick={e => e.stopPropagation()}>
        <div className="hiw-header">
          <h2>How It Works</h2>
          <button className="hiw-close" onClick={onClose} aria-label="Close">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" strokeWidth={2} stroke="currentColor" width="20" height="20">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="hiw-body">

          <section className="hiw-section">
            <h3>🌍 What is RCMES Climate Assistant?</h3>
            <p>
              RCMES (Regional Climate Model Evaluation System) is a NASA climate analysis platform.
              This assistant gives you a conversational interface to <strong>38 TB of NASA's NEX-GDDP-CMIP6 climate projections</strong> —
              global daily downscaled data at 0.25° resolution covering 1950–2100.
            </p>
          </section>

          <section className="hiw-section">
            <h3>🔄 Architecture</h3>
            <div className="hiw-steps">
              <div className="hiw-step">
                <span className="hiw-step-num">1</span>
                <div>
                  <strong>You ask a question</strong>
                  <p>Type a natural language request — e.g. "Show temperature trends for California under SSP5-8.5"</p>
                </div>
              </div>
              <div className="hiw-step">
                <span className="hiw-step-num">2</span>
                <div>
                  <strong>AI selects tools</strong>
                  <p>An Azure OpenAI (GPT-4o) model decides which climate tools to call and in what order, using the Model Context Protocol (MCP).</p>
                </div>
              </div>
              <div className="hiw-step">
                <span className="hiw-step-num">3</span>
                <div>
                  <strong>Data is streamed from AWS S3</strong>
                  <p>Climate data is loaded on-demand from NASA's NEX-GDDP-CMIP6 Zarr store on Amazon S3 — no full download needed.</p>
                </div>
              </div>
              <div className="hiw-step">
                <span className="hiw-step-num">4</span>
                <div>
                  <strong>Analysis &amp; visualization</strong>
                  <p>Tools process the data with xarray/Dask and generate maps, time-series plots, trend lines, and climate indices.</p>
                </div>
              </div>
              <div className="hiw-step">
                <span className="hiw-step-num">5</span>
                <div>
                  <strong>Push to MMGIS (optional)</strong>
                  <p>Results can be exported as Cloud-Optimized GeoTIFFs and pushed as live map layers to NASA's MMGIS geospatial platform.</p>
                </div>
              </div>
            </div>
          </section>

          <section className="hiw-section">
            <h3>🛠️ Available Tools</h3>
            <div className="hiw-tools-grid">
              {[
                { icon: '📥', label: 'Data Access', desc: 'Load CMIP6 climate data by model, scenario, region, variable, and time range' },
                { icon: '✂️', label: 'Processing', desc: 'Temporal/spatial subsetting, regridding, and unit conversion' },
                { icon: '📊', label: 'Analysis', desc: 'Statistics, trend detection, climatology, and ensemble comparisons' },
                { icon: '🌡️', label: 'Climate Indices', desc: 'ETCCDI indices, heatwaves, drought metrics, and extreme event analysis' },
                { icon: '🗺️', label: 'Visualization', desc: 'Maps, time-series plots, Taylor diagrams, Hovmöller diagrams' },
                { icon: '📡', label: 'MMGIS Push', desc: 'Export GeoTIFFs and publish live map layers to NASA MMGIS' },
              ].map(t => (
                <div key={t.label} className="hiw-tool-card">
                  <span className="hiw-tool-icon">{t.icon}</span>
                  <strong>{t.label}</strong>
                  <p>{t.desc}</p>
                </div>
              ))}
            </div>
          </section>

          <section className="hiw-section">
            <h3>📡 Data Source</h3>
            <p>
              <strong>NASA NEX-GDDP-CMIP6</strong> — 35 climate models × 5 scenarios (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5, historical),
              9 variables (temperature, precipitation, humidity, radiation, wind), 1950–2100.
              Hosted as Zarr on <strong>AWS S3 (us-west-2)</strong>, streamed without full download.
            </p>
          </section>

          <section className="hiw-section hiw-section--links">
            <a href="https://github.com/yunks128/rcmes-mcp" target="_blank" rel="noopener noreferrer">
              <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
              View on GitHub
            </a>
            <a href="https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6" target="_blank" rel="noopener noreferrer">
              🌐 NEX-GDDP-CMIP6 Dataset
            </a>
            <a href="/mmgis/?mission=climate" target="_blank" rel="noopener noreferrer">
              🗺️ Open MMGIS Map
            </a>
          </section>

        </div>
      </div>
    </div>
  );
}
