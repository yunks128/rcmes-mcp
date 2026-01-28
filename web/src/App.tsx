import { useState, useEffect, useCallback } from 'react';
import * as api from './api';
import Architecture from './Architecture';

// Preset regions
const REGIONS: Record<string, { lat_min: number; lat_max: number; lon_min: number; lon_max: number }> = {
  California: { lat_min: 32.0, lat_max: 42.0, lon_min: -124.0, lon_max: -114.0 },
  Texas: { lat_min: 25.5, lat_max: 36.5, lon_min: -106.5, lon_max: -93.5 },
  Florida: { lat_min: 24.5, lat_max: 31.0, lon_min: -87.5, lon_max: -80.0 },
  Global: { lat_min: -60.0, lat_max: 90.0, lon_min: -180.0, lon_max: 180.0 },
};

interface Message {
  role: 'user' | 'assistant';
  content: string;
  image?: string;
}

type Page = 'explorer' | 'architecture';

function App() {
  // Navigation state
  const [currentPage, setCurrentPage] = useState<Page>('explorer');

  // Metadata state
  const [models, setModels] = useState<string[]>([]);
  const [variables, setVariables] = useState<api.Variable[]>([]);
  const [scenarios, setScenarios] = useState<api.Scenario[]>([]);
  const [datasets, setDatasets] = useState<api.Dataset[]>([]);

  // Form state
  const [selectedModel, setSelectedModel] = useState('ACCESS-CM2');
  const [selectedVariable, setSelectedVariable] = useState('tasmax');
  const [selectedScenario, setSelectedScenario] = useState('ssp585');
  const [selectedRegion, setSelectedRegion] = useState('California');
  const [latMin, setLatMin] = useState(32.0);
  const [latMax, setLatMax] = useState(42.0);
  const [lonMin, setLonMin] = useState(-124.0);
  const [lonMax, setLonMax] = useState(-114.0);
  const [startDate, setStartDate] = useState('2050-01-01');
  const [endDate, setEndDate] = useState('2050-12-31');

  // Analysis state
  const [selectedAnalysis, setSelectedAnalysis] = useState('statistics');
  const [selectedViz, setSelectedViz] = useState('map');

  // App state
  const [currentDatasetId, setCurrentDatasetId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load metadata on mount
  useEffect(() => {
    async function loadMetadata() {
      try {
        const [modelsRes, varsRes, scenariosRes] = await Promise.all([
          api.listModels(),
          api.listVariables(),
          api.listScenarios(),
        ]);
        setModels(modelsRes.models?.slice(0, 15) || []);
        setVariables(varsRes.variables || []);
        setScenarios(scenariosRes.scenarios || []);
      } catch (err) {
        console.error('Failed to load metadata:', err);
      }
    }
    loadMetadata();
  }, []);

  // Refresh datasets list
  const refreshDatasets = useCallback(async () => {
    try {
      const result = await api.listDatasets();
      setDatasets(result.datasets || []);
    } catch (err) {
      console.error('Failed to load datasets:', err);
    }
  }, []);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  // Handle region preset change
  const handleRegionChange = (region: string) => {
    setSelectedRegion(region);
    if (region !== 'Custom' && REGIONS[region]) {
      const r = REGIONS[region];
      setLatMin(r.lat_min);
      setLatMax(r.lat_max);
      setLonMin(r.lon_min);
      setLonMax(r.lon_max);
    }
  };

  // Add message to chat
  const addMessage = (role: 'user' | 'assistant', content: string, image?: string) => {
    setMessages((prev) => [...prev, { role, content, image }]);
  };

  // Load data with auto-visualization
  const handleLoadData = async () => {
    setLoading(true);
    setError(null);
    setMessages([]); // Clear previous messages

    try {
      // Load the data
      const result = await api.loadData({
        variable: selectedVariable,
        model: selectedModel,
        scenario: selectedScenario,
        start_date: startDate,
        end_date: endDate,
        lat_min: latMin,
        lat_max: latMax,
        lon_min: lonMin,
        lon_max: lonMax,
      });

      const datasetId = result.dataset_id;
      setCurrentDatasetId(datasetId);
      await refreshDatasets();

      const dims = result.dimensions;

      // Add data summary message
      let summaryMsg = `## Data Loaded Successfully\n\n`;
      summaryMsg += `**Variable:** ${selectedVariable}\n`;
      summaryMsg += `**Model:** ${selectedModel}\n`;
      summaryMsg += `**Scenario:** ${selectedScenario}\n`;
      summaryMsg += `**Region:** ${selectedRegion} (${latMin}°-${latMax}°N, ${lonMin}°-${lonMax}°E)\n`;
      summaryMsg += `**Period:** ${startDate} to ${endDate}\n\n`;
      summaryMsg += `**Dimensions:** ${dims.time} time steps, ${dims.lat}×${dims.lon} grid\n`;
      summaryMsg += `**Dataset ID:** \`${datasetId}\``;
      addMessage('assistant', summaryMsg);

      // Auto-generate statistics
      try {
        const statsResult = await api.analyze(datasetId, 'statistics');
        let statsMsg = `## Statistics\n\n`;
        statsMsg += `| Metric | Value |\n|--------|-------|\n`;
        statsMsg += `| Mean | ${statsResult.mean?.toFixed(2)} |\n`;
        statsMsg += `| Std Dev | ${statsResult.std?.toFixed(2)} |\n`;
        statsMsg += `| Min | ${statsResult.min?.toFixed(2)} |\n`;
        statsMsg += `| Max | ${statsResult.max?.toFixed(2)} |\n`;
        if (statsResult.percentiles) {
          const p = statsResult.percentiles;
          statsMsg += `| Median | ${p.p50?.toFixed(2)} |\n`;
          statsMsg += `| 5th-95th | ${p.p5?.toFixed(2)} - ${p.p95?.toFixed(2)} |\n`;
        }
        addMessage('assistant', statsMsg);
      } catch (err) {
        console.error('Failed to get statistics:', err);
      }

      // Auto-generate map visualization
      try {
        const title = `${selectedVariable} - ${selectedModel} (${selectedScenario})`;
        const mapResult = await api.visualize(datasetId, 'map', title);
        if (mapResult.image_base64) {
          addMessage('assistant', `## Spatial Map`, mapResult.image_base64);
        }
      } catch (err) {
        console.error('Failed to generate map:', err);
      }

      // Auto-generate time series
      try {
        const title = `${selectedVariable} Time Series - ${selectedRegion}`;
        const tsResult = await api.visualize(datasetId, 'timeseries', title, true);
        if (tsResult.image_base64) {
          addMessage('assistant', `## Time Series`, tsResult.image_base64);
        }
      } catch (err) {
        console.error('Failed to generate time series:', err);
      }

    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Failed to load data';
      setError(errMsg);
      addMessage('assistant', `Error: ${errMsg}`);
    } finally {
      setLoading(false);
    }
  };

  // Run analysis
  const handleAnalysis = async () => {
    if (!currentDatasetId) {
      setError('Please load data first');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await api.analyze(currentDatasetId, selectedAnalysis);

      if (result.dataset_id) {
        setCurrentDatasetId(result.dataset_id);
        await refreshDatasets();
      }

      let msg = '';
      if (selectedAnalysis === 'statistics') {
        msg = `**Statistics for ${result.variable || 'Data'}:**\n\n`;
        msg += `| Metric | Value |\n|--------|-------|\n`;
        msg += `| Mean | ${result.mean?.toFixed(2)} |\n`;
        msg += `| Std Dev | ${result.std?.toFixed(2)} |\n`;
        msg += `| Min | ${result.min?.toFixed(2)} |\n`;
        msg += `| Max | ${result.max?.toFixed(2)} |\n`;
        if (result.percentiles) {
          const p = result.percentiles;
          msg += `| 5th Percentile | ${p.p5?.toFixed(2)} |\n`;
          msg += `| 25th Percentile | ${p.p25?.toFixed(2)} |\n`;
          msg += `| Median (50th) | ${p.p50?.toFixed(2)} |\n`;
          msg += `| 75th Percentile | ${p.p75?.toFixed(2)} |\n`;
          msg += `| 95th Percentile | ${p.p95?.toFixed(2)} |\n`;
        }
      } else if (selectedAnalysis === 'trend') {
        msg = `**Trend Analysis for ${result.variable || 'Data'}:**\n\n`;
        msg += `| Metric | Value |\n|--------|-------|\n`;
        const trend = result.trend;
        if (trend?.slope_per_decade !== undefined) {
          msg += `| Trend | ${trend.slope_per_decade.toFixed(4)} ${trend.unit || 'per decade'} |\n`;
        }
        if (trend?.p_value !== undefined) {
          const sig = trend.p_value < 0.05 ? 'Yes' : 'No';
          msg += `| P-value | ${trend.p_value.toExponential(2)} |\n`;
          msg += `| Significant (p<0.05) | ${sig} |\n`;
        }
        if (trend?.r_squared !== undefined) {
          msg += `| R-squared | ${trend.r_squared.toFixed(4)} |\n`;
        }
        if (result.interpretation) {
          msg += `\n${result.interpretation}`;
        }
      } else if (selectedAnalysis === 'heatwaves') {
        msg = `**Heatwave Analysis:**\n\n`;
        if (result.summary) {
          const s = result.summary;
          msg += `| Metric | Value |\n|--------|-------|\n`;
          if (s.mean_annual_hot_days !== undefined) {
            msg += `| Hot Days (>90th percentile) | ${s.mean_annual_hot_days.toFixed(1)} days/year |\n`;
          }
          if (s.mean_annual_heatwave_frequency !== undefined) {
            msg += `| Heatwave Events | ${s.mean_annual_heatwave_frequency.toFixed(1)} per year |\n`;
          }
          if (s.mean_heatwave_duration !== undefined) {
            msg += `| Avg Duration | ${s.mean_heatwave_duration.toFixed(1)} days |\n`;
          }
        }
        msg += `\nDataset ID: \`${result.dataset_id}\``;
      } else {
        msg = `**${selectedAnalysis} Results:**\n\nCompleted successfully. New dataset ID: \`${result.dataset_id || 'N/A'}\``;
      }

      addMessage('assistant', msg);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Analysis failed';
      setError(errMsg);
      addMessage('assistant', `Error: ${errMsg}`);
    } finally {
      setLoading(false);
    }
  };

  // Generate visualization
  const handleVisualize = async () => {
    if (!currentDatasetId) {
      setError('Please load data first');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const title = `${selectedVariable} - ${selectedModel} (${selectedScenario})`;
      const result = await api.visualize(
        currentDatasetId,
        selectedViz,
        title,
        selectedViz === 'timeseries'
      );

      if (result.image_base64) {
        addMessage('assistant', `Generated ${selectedViz}:`, result.image_base64);
      } else {
        addMessage('assistant', 'Visualization generated (no image returned)');
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Visualization failed';
      setError(errMsg);
      addMessage('assistant', `Error: ${errMsg}`);
    } finally {
      setLoading(false);
    }
  };

  // Navigation component
  const NavBar = () => (
    <nav className="top-nav">
      <button
        className={currentPage === 'explorer' ? 'active' : ''}
        onClick={() => setCurrentPage('explorer')}
      >
        Data Explorer
      </button>
      <button
        className={currentPage === 'architecture' ? 'active' : ''}
        onClick={() => setCurrentPage('architecture')}
      >
        Architecture
      </button>
    </nav>
  );

  // Render Architecture page
  if (currentPage === 'architecture') {
    return (
      <div className="app-container">
        <NavBar />
        <main className="main-content full-width">
          <Architecture />
        </main>
      </div>
    );
  }

  // Render Data Explorer page
  return (
    <div className="app-container">
      <NavBar />

      {/* Sidebar */}
      <aside className="sidebar">
        <h1>RCMES Explorer</h1>
        <p className="subtitle">NASA's NEX-GDDP-CMIP6 Climate Data</p>

        <h2>Data Selection</h2>

        <div className="form-group">
          <label>Climate Model</label>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Variable</label>
          <select value={selectedVariable} onChange={(e) => setSelectedVariable(e.target.value)}>
            {variables.map((v) => (
              <option key={v.name} value={v.name}>
                {v.name} - {v.long_name}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Emissions Scenario</label>
          <select value={selectedScenario} onChange={(e) => setSelectedScenario(e.target.value)}>
            {scenarios.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id} - {s.description.slice(0, 30)}...
              </option>
            ))}
          </select>
        </div>

        <div className="divider" />

        <h2>Region</h2>

        <div className="form-group">
          <label>Preset Region</label>
          <select value={selectedRegion} onChange={(e) => handleRegionChange(e.target.value)}>
            {[...Object.keys(REGIONS), 'Custom'].map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>

        <div className="row">
          <div className="form-group">
            <label>Lat Min</label>
            <input
              type="number"
              value={latMin}
              onChange={(e) => setLatMin(parseFloat(e.target.value))}
              min={-60}
              max={90}
              step={0.1}
            />
          </div>
          <div className="form-group">
            <label>Lat Max</label>
            <input
              type="number"
              value={latMax}
              onChange={(e) => setLatMax(parseFloat(e.target.value))}
              min={-60}
              max={90}
              step={0.1}
            />
          </div>
        </div>

        <div className="row">
          <div className="form-group">
            <label>Lon Min</label>
            <input
              type="number"
              value={lonMin}
              onChange={(e) => setLonMin(parseFloat(e.target.value))}
              min={-180}
              max={180}
              step={0.1}
            />
          </div>
          <div className="form-group">
            <label>Lon Max</label>
            <input
              type="number"
              value={lonMax}
              onChange={(e) => setLonMax(parseFloat(e.target.value))}
              min={-180}
              max={180}
              step={0.1}
            />
          </div>
        </div>

        <div className="divider" />

        <h2>Time Period</h2>

        <div className="row">
          <div className="form-group">
            <label>Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
        </div>

        <button
          className="btn-primary"
          onClick={handleLoadData}
          disabled={loading}
          style={{ marginTop: '16px' }}
        >
          {loading ? <span className="spinner" /> : 'Load Data'}
        </button>

        <div className="divider" />

        <h2>Analysis</h2>

        <div className="form-group">
          <label>Analysis Type</label>
          <select value={selectedAnalysis} onChange={(e) => setSelectedAnalysis(e.target.value)}>
            <option value="statistics">Statistics</option>
            <option value="trend">Trend</option>
            <option value="climatology">Climatology</option>
            <option value="heatwaves">Heatwaves</option>
            <option value="regional_mean">Regional Mean</option>
          </select>
        </div>

        <button
          className="btn-secondary"
          onClick={handleAnalysis}
          disabled={loading || !currentDatasetId}
          style={{ width: '100%' }}
        >
          Run Analysis
        </button>

        <div className="divider" />

        <h2>Visualization</h2>

        <div className="form-group">
          <label>Plot Type</label>
          <select value={selectedViz} onChange={(e) => setSelectedViz(e.target.value)}>
            <option value="map">Map</option>
            <option value="timeseries">Time Series</option>
            <option value="histogram">Histogram</option>
          </select>
        </div>

        <button
          className="btn-secondary"
          onClick={handleVisualize}
          disabled={loading || !currentDatasetId}
          style={{ width: '100%' }}
        >
          Generate Plot
        </button>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <div className="results-area">
          <div className="results-main">
            <div className="card">
              <h2 style={{ marginTop: 0 }}>Results</h2>

              {error && <div className="alert error">{error}</div>}

              {loading && (
                <div className="loading-overlay">
                  <div className="spinner large" />
                  <p>Loading and analyzing data...</p>
                </div>
              )}

              {messages.length === 0 && !loading ? (
                <div className="empty-state">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"
                    />
                  </svg>
                  <p>Select data parameters and click "Load Data" to begin</p>
                  <p className="hint">The system will automatically generate statistics, map, and time series plots</p>
                </div>
              ) : (
                <div className="messages">
                  {messages.map((msg, i) => (
                    <div key={i} className={`message ${msg.role}`}>
                      <div dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }} />
                      {msg.image && (
                        <img
                          src={`data:image/png;base64,${msg.image}`}
                          alt="Visualization"
                        />
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="results-sidebar">
            <h3>Loaded Datasets</h3>
            {datasets.length === 0 ? (
              <p className="empty-state" style={{ padding: '20px 0' }}>
                No datasets loaded
              </p>
            ) : (
              datasets.map((ds) => (
                <div
                  key={ds.id}
                  className={`dataset-item ${ds.id === currentDatasetId ? 'selected' : ''}`}
                  onClick={() => setCurrentDatasetId(ds.id)}
                >
                  <code>{ds.id.slice(0, 8)}...</code>
                  <div className="meta">
                    {ds.variable} | {ds.model} | {ds.scenario}
                  </div>
                </div>
              ))
            )}

            {currentDatasetId && (
              <>
                <div className="divider" />
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <strong>Current:</strong>
                  <br />
                  <code>{currentDatasetId}</code>
                </p>
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

// Markdown formatter with proper table support
function formatMarkdown(text: string): string {
  // First, handle tables
  const lines = text.split('\n');
  const result: string[] = [];
  let inTable = false;
  let tableRows: string[] = [];

  for (const line of lines) {
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      // This is a table row
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      // Skip separator rows (|---|---|)
      if (!line.match(/^\|[\s-:|]+\|$/)) {
        const cells = line.split('|').filter(Boolean);
        if (tableRows.length === 0) {
          // Header row
          tableRows.push(`<tr>${cells.map(c => `<th>${c.trim()}</th>`).join('')}</tr>`);
        } else {
          // Data row
          tableRows.push(`<tr>${cells.map(c => `<td>${c.trim()}</td>`).join('')}</tr>`);
        }
      }
    } else {
      // Not a table row
      if (inTable) {
        // Close the table
        result.push(`<table class="stats-table">${tableRows.join('')}</table>`);
        inTable = false;
        tableRows = [];
      }
      result.push(line);
    }
  }

  // Close any remaining table
  if (inTable && tableRows.length > 0) {
    result.push(`<table class="stats-table">${tableRows.join('')}</table>`);
  }

  // Join and apply other formatting
  return result.join('\n')
    .replace(/## (.*?)(\n|$)/g, '<h3>$1</h3>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br />');
}

export default App;
