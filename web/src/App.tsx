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

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

type Page = 'explorer' | 'architecture';

function App() {
  // Navigation state
  const [currentPage, setCurrentPage] = useState<Page>('explorer');

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  // Metadata state
  const [models, setModels] = useState<string[]>([]);
  const [variables, setVariables] = useState<api.Variable[]>([]);
  const [scenarios, setScenarios] = useState<api.Scenario[]>([]);
  const [datasets, setDatasets] = useState<api.Dataset[]>([]);

  // Form state
  const [selectedModel, setSelectedModel] = useState('ACCESS-CM2');
  const [selectedVariable, setSelectedVariable] = useState('tasmax');
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>(['ssp585']);
  const [allDatasetIds, setAllDatasetIds] = useState<string[]>([]);
  const [scenarioLabels, setScenarioLabels] = useState<string[]>([]);
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
  const [secondDatasetId, setSecondDatasetId] = useState<string | null>(null);
  const [correlationType, setCorrelationType] = useState<'temporal' | 'spatial'>('temporal');

  // Download state
  const [downloadFormat, setDownloadFormat] = useState<'netcdf' | 'csv'>('netcdf');
  const [downloading, setDownloading] = useState(false);

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
    if (selectedScenarios.length === 0) {
      setError('Please select at least one emissions scenario');
      return;
    }

    setLoading(true);
    setError(null);
    setMessages([]);

    try {
      // Load data for each selected scenario
      const loadedIds: string[] = [];
      const labels: string[] = [];

      for (const scenario of selectedScenarios) {
        const result = await api.loadData({
          variable: selectedVariable,
          model: selectedModel,
          scenario,
          start_date: startDate,
          end_date: endDate,
          lat_min: latMin,
          lat_max: latMax,
          lon_min: lonMin,
          lon_max: lonMax,
        });
        loadedIds.push(result.dataset_id);
        labels.push(scenario);

        const dims = result.dimensions;
        let summaryMsg = `## Data Loaded: ${scenario}\n\n`;
        summaryMsg += `**Variable:** ${selectedVariable}\n`;
        summaryMsg += `**Model:** ${selectedModel}\n`;
        summaryMsg += `**Scenario:** ${scenario}\n`;
        summaryMsg += `**Region:** ${selectedRegion} (${latMin}°-${latMax}°N, ${lonMin}°-${lonMax}°E)\n`;
        summaryMsg += `**Period:** ${startDate} to ${endDate}\n\n`;
        summaryMsg += `**Dimensions:** ${dims.time} time steps, ${dims.lat}×${dims.lon} grid\n`;
        summaryMsg += `**Dataset ID:** \`${result.dataset_id}\``;
        addMessage('assistant', summaryMsg);
      }

      // Set the first loaded dataset as current, track all
      setCurrentDatasetId(loadedIds[0]);
      setAllDatasetIds(loadedIds);
      setScenarioLabels(labels);
      await refreshDatasets();

      // Auto-generate statistics for each scenario
      for (let i = 0; i < loadedIds.length; i++) {
        try {
          const statsResult = await api.analyze(loadedIds[i], 'statistics');
          let statsMsg = `## Statistics — ${labels[i]}\n\n`;
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
          console.error(`Failed to get statistics for ${labels[i]}:`, err);
        }
      }

      // Auto-generate map visualization
      if (loadedIds.length >= 2) {
        // Comparison map for multiple scenarios
        try {
          const title = `${selectedVariable} - ${selectedModel} (${labels.join(' vs ')})`;
          const mapResult = await api.visualize(loadedIds[0], 'map', title, false, loadedIds, labels);
          if (mapResult.image_base64) {
            addMessage('assistant', `## Scenario Comparison Map`, mapResult.image_base64);
          }
        } catch (err) {
          console.error('Failed to generate comparison map:', err);
        }
      } else {
        try {
          const title = `${selectedVariable} - ${selectedModel} (${labels[0]})`;
          const mapResult = await api.visualize(loadedIds[0], 'map', title);
          if (mapResult.image_base64) {
            addMessage('assistant', `## Spatial Map`, mapResult.image_base64);
          }
        } catch (err) {
          console.error('Failed to generate map:', err);
        }
      }

      // Auto-generate comparison time series (all scenarios overlaid)
      try {
        const title = `${selectedVariable} Time Series - ${selectedRegion} (${labels.join(', ')})`;
        const tsResult = await api.visualize(
          loadedIds[0], 'timeseries', title, true, loadedIds, labels
        );
        if (tsResult.image_base64) {
          const heading = loadedIds.length >= 2 ? '## Scenario Comparison Time Series' : '## Time Series';
          addMessage('assistant', heading, tsResult.image_base64);
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

    // Check for correlation - needs second dataset
    if (selectedAnalysis === 'correlation') {
      if (!secondDatasetId) {
        setError('Please select a second dataset for correlation');
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const result = await api.calculateCorrelation(currentDatasetId, secondDatasetId, correlationType);
        if (result.dataset_id) {
          setCurrentDatasetId(result.dataset_id);
          await refreshDatasets();
        }
        const msg = `**Correlation Analysis:**\n\n| Metric | Value |\n|--------|-------|\n| Type | ${result.correlation_type} |\n| Mean Correlation | ${result.mean_correlation?.toFixed(4)} |\n\nResult dataset: \`${result.dataset_id}\``;
        addMessage('assistant', msg);
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : 'Correlation failed';
        setError(errMsg);
        addMessage('assistant', `Error: ${errMsg}`);
      } finally {
        setLoading(false);
      }
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
      const scenarioStr = scenarioLabels.length > 1
        ? scenarioLabels.join(', ')
        : scenarioLabels[0] || selectedScenarios[0] || '';
      const title = `${selectedVariable} - ${selectedModel} (${scenarioStr})`;

      // Use all loaded dataset IDs for multi-scenario comparison
      const ids = allDatasetIds.length > 0 ? allDatasetIds : [currentDatasetId];
      const lbls = scenarioLabels.length > 0 ? scenarioLabels : undefined;

      const result = await api.visualize(
        currentDatasetId,
        selectedViz,
        title,
        selectedViz === 'timeseries',
        ids.length > 1 ? ids : undefined,
        ids.length > 1 ? lbls : undefined
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

  // Download dataset
  const handleDownload = async () => {
    if (!currentDatasetId) {
      setError('Please load data first');
      return;
    }

    setDownloading(true);
    try {
      await api.downloadDataset(currentDatasetId, downloadFormat);
      addMessage('assistant', `Downloaded dataset as ${downloadFormat === 'netcdf' ? 'NetCDF (.nc)' : 'CSV (.csv)'}`);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Download failed';
      setError(errMsg);
    } finally {
      setDownloading(false);
    }
  };

  // Send chat message
  const handleSendChat = async () => {
    if (!chatInput.trim()) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: chatInput,
      timestamp: new Date(),
    };
    setChatMessages((prev) => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);

    try {
      // Send to chat API
      const response = await api.sendChatMessage(chatInput, currentDatasetId || undefined);

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
      };
      setChatMessages((prev) => [...prev, assistantMessage]);

      // If the response includes an action result, add it to the main messages
      if (response.action_result) {
        if (response.action_result.image_base64) {
          addMessage('assistant', response.action_result.message || 'Generated visualization:', response.action_result.image_base64);
        } else if (response.action_result.dataset_id) {
          setCurrentDatasetId(response.action_result.dataset_id);
          await refreshDatasets();
          addMessage('assistant', response.action_result.message || `Analysis complete. Dataset ID: ${response.action_result.dataset_id}`);
        }
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Failed to send message';
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errMsg}`,
        timestamp: new Date(),
      };
      setChatMessages((prev) => [...prev, errorMessage]);
    } finally {
      setChatLoading(false);
    }
  };

  // Handle chat input key press
  const handleChatKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendChat();
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
          <label>Emissions Scenarios</label>
          <div className="checkbox-group">
            {scenarios.map((s) => (
              <label key={s.id} className="checkbox-label">
                <input
                  type="checkbox"
                  checked={selectedScenarios.includes(s.id)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedScenarios((prev) => [...prev, s.id]);
                    } else {
                      setSelectedScenarios((prev) => prev.filter((id) => id !== s.id));
                    }
                  }}
                />
                <span className="checkbox-text">{s.id}</span>
                <span className="checkbox-desc">{s.description.slice(0, 25)}</span>
              </label>
            ))}
          </div>
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
          disabled={loading || selectedScenarios.length === 0}
          style={{ marginTop: '16px' }}
        >
          {loading ? <span className="spinner" /> : selectedScenarios.length > 1
            ? `Load Data (${selectedScenarios.length} scenarios)`
            : 'Load Data'}
        </button>

        <div className="divider" />

        <h2>Download Data</h2>

        {currentDatasetId ? (
          <>
            <div className="form-group">
              <label>Format</label>
              <select value={downloadFormat} onChange={(e) => setDownloadFormat(e.target.value as 'netcdf' | 'csv')}>
                <option value="netcdf">NetCDF (.nc)</option>
                <option value="csv">CSV (.csv)</option>
              </select>
            </div>
            <button
              className="btn-secondary"
              onClick={handleDownload}
              disabled={downloading}
              style={{ width: '100%' }}
            >
              {downloading ? <span className="spinner" /> : 'Download'}
            </button>
          </>
        ) : (
          <p className="empty-hint">Load data first to enable download</p>
        )}

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
            <option value="correlation">Correlation</option>
          </select>
        </div>

        {selectedAnalysis === 'correlation' && (
          <>
            <div className="form-group">
              <label>Compare with Dataset</label>
              {datasets.length < 2 ? (
                <p className="empty-hint" style={{ color: '#f59e0b', fontSize: '12px' }}>
                  Load at least 2 datasets for correlation
                </p>
              ) : (
                <select
                  value={secondDatasetId || ''}
                  onChange={(e) => setSecondDatasetId(e.target.value || null)}
                >
                  <option value="">Select dataset...</option>
                  {datasets
                    .filter((ds) => ds.id !== currentDatasetId)
                    .map((ds) => (
                      <option key={ds.id} value={ds.id}>
                        {ds.id} ({ds.variable || 'N/A'})
                      </option>
                    ))}
                </select>
              )}
            </div>
            <div className="form-group">
              <label>Correlation Type</label>
              <select
                value={correlationType}
                onChange={(e) => setCorrelationType(e.target.value as 'temporal' | 'spatial')}
              >
                <option value="temporal">Temporal (over time)</option>
                <option value="spatial">Spatial (over space)</option>
              </select>
            </div>
          </>
        )}

        <button
          className="btn-secondary"
          onClick={handleAnalysis}
          disabled={loading || !currentDatasetId || (selectedAnalysis === 'correlation' && !secondDatasetId)}
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
        <div className="explorer-layout">
          {/* Main Content Column */}
          <div className="main-column">
            {/* Loaded Datasets Section */}
            <div className="datasets-section">
              <h3>Loaded Datasets</h3>
              <div className="datasets-list">
                {datasets.length === 0 ? (
                  <p className="empty-hint">No datasets loaded</p>
                ) : (
                  datasets.map((ds) => (
                    <div
                      key={ds.id}
                      className={`dataset-item ${ds.id === currentDatasetId ? 'selected' : ''}`}
                      onClick={() => setCurrentDatasetId(ds.id)}
                    >
                      <div className="dataset-detail"><span className="label">ID:</span> <code>{ds.id}</code></div>
                      <div className="dataset-detail"><span className="label">Variable:</span> {ds.variable || 'N/A'}</div>
                      <div className="dataset-detail"><span className="label">Model:</span> {ds.model || 'N/A'}</div>
                      <div className="dataset-detail"><span className="label">Scenario:</span> {ds.scenario || 'N/A'}</div>
                      {ds.time_range && (
                        <div className="dataset-detail"><span className="label">Time:</span> {ds.time_range[0]} to {ds.time_range[1]}</div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Results Section */}
            <div className="card results-card">
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

          {/* Chat Panel */}
          <div className="chat-panel">
            <div className="chat-header">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" width="20" height="20">
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H8.25m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H12m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 0 1-2.555-.337A5.972 5.972 0 0 1 5.41 20.97a5.969 5.969 0 0 1-.474-.065 4.48 4.48 0 0 0 .978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25Z" />
              </svg>
              <h3>RCMES Copilot</h3>
            </div>

            <div className="chat-messages">
              {chatMessages.length === 0 ? (
                <div className="chat-welcome">
                  <div className="chat-welcome-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
                    </svg>
                  </div>
                  <p className="chat-welcome-title">RCMES Copilot</p>
                  <p className="chat-welcome-hint">Ask questions about climate data, request analyses, or get help understanding your results.</p>
                  <div className="chat-suggestions">
                    <button onClick={() => setChatInput('What is the temperature trend?')}>Temperature trend?</button>
                    <button onClick={() => setChatInput('Show me a map of the data')}>Show map</button>
                    <button onClick={() => setChatInput('What are the statistics?')}>Get statistics</button>
                  </div>
                </div>
              ) : (
                chatMessages.map((msg, i) => (
                  <div key={i} className={`chat-message ${msg.role}`}>
                    <div className="chat-message-content">
                      {msg.content}
                    </div>
                    <div className="chat-message-time">
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                ))
              )}
              {chatLoading && (
                <div className="chat-message assistant">
                  <div className="chat-typing">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}
            </div>

            {/* Always show suggestion buttons */}
            <div className="chat-quick-actions">
              <button onClick={() => setChatInput('Show statistics')} disabled={chatLoading}>Statistics</button>
              <button onClick={() => setChatInput('Analyze the trend')} disabled={chatLoading}>Trend</button>
              <button onClick={() => setChatInput('Generate a map')} disabled={chatLoading}>Map</button>
              <button onClick={() => setChatInput('Show time series')} disabled={chatLoading}>Time Series</button>
            </div>

            <div className="chat-input-area">
              <textarea
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={handleChatKeyPress}
                placeholder="Ask about climate data..."
                rows={2}
                disabled={chatLoading}
              />
              <button
                onClick={handleSendChat}
                disabled={chatLoading || !chatInput.trim()}
                className="chat-send-btn"
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" width="20" height="20">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
                </svg>
              </button>
            </div>
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
