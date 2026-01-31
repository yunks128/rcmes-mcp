/**
 * RCMES API Client
 *
 * Functions for communicating with the FastAPI backend.
 */

const API_BASE = '/api';

export interface Variable {
  name: string;
  long_name: string;
  units: string;
}

export interface Scenario {
  id: string;
  description: string;
}

export interface Dataset {
  id: string;
  variable?: string;
  model?: string;
  scenario?: string;
  time_range?: [string, string];
}

export interface LoadDataParams {
  variable: string;
  model: string;
  scenario: string;
  start_date: string;
  end_date: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface AnalysisResult {
  dataset_id?: string;
  variable?: string;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  percentiles?: {
    p5?: number;
    p25?: number;
    p50?: number;
    p75?: number;
    p95?: number;
  };
  trend?: {
    slope_per_decade?: number;
    unit?: string;
    p_value?: number;
    r_squared?: number;
  };
  interpretation?: string;
  summary?: {
    mean_annual_hot_days?: number;
    mean_annual_heatwave_frequency?: number;
    mean_heatwave_duration?: number;
  };
  error?: string;
}

export interface VisualizationResult {
  image_base64?: string;
  error?: string;
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || 'Request failed');
  }

  return response.json();
}

// Metadata endpoints
export async function listModels(): Promise<{ models: string[] }> {
  return fetchJson(`${API_BASE}/models`);
}

export async function listVariables(): Promise<{ variables: Variable[] }> {
  return fetchJson(`${API_BASE}/variables`);
}

export async function listScenarios(): Promise<{ scenarios: Scenario[] }> {
  return fetchJson(`${API_BASE}/scenarios`);
}

export async function listDatasets(): Promise<{ datasets: Dataset[]; dataset_count: number }> {
  return fetchJson(`${API_BASE}/datasets`);
}

// Data loading
export async function loadData(params: LoadDataParams): Promise<{
  dataset_id: string;
  dimensions: { time: number; lat: number; lon: number };
}> {
  return fetchJson(`${API_BASE}/load-data`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

// Analysis
export async function analyze(
  dataset_id: string,
  analysis_type: string
): Promise<AnalysisResult> {
  return fetchJson(`${API_BASE}/analyze`, {
    method: 'POST',
    body: JSON.stringify({ dataset_id, analysis_type }),
  });
}

// Visualization
export async function visualize(
  dataset_id: string,
  viz_type: string,
  title?: string,
  show_trend = false
): Promise<VisualizationResult> {
  return fetchJson(`${API_BASE}/visualize`, {
    method: 'POST',
    body: JSON.stringify({ dataset_id, viz_type, title, show_trend }),
  });
}

// Health check
export async function healthCheck(): Promise<{ status: string }> {
  return fetchJson(`${API_BASE}/health`);
}

// Chat
export interface ChatResponse {
  response: string;
  action_result?: {
    dataset_id?: string;
    image_base64?: string;
    message?: string;
  };
}

export async function sendChatMessage(
  message: string,
  dataset_id?: string
): Promise<ChatResponse> {
  return fetchJson(`${API_BASE}/chat`, {
    method: 'POST',
    body: JSON.stringify({ message, dataset_id }),
  });
}
