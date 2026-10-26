/** REST API client ( Axios) for the FastAPI backend. */
import axios from 'axios';
import type {
  ClassInfo,
  EcgEvent,
  EcgSendResponse,
  Explanation,
  ExperimentsResponse,
  Prediction,
  SimulationStatus,
  Statistics,
  SystemStatus,
} from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '',
  timeout: 15_000,
});

export async function getHealth(): Promise<Record<string, unknown>> {
  const { data } = await api.get('/health');
  return data;
}

export async function getSystemStatus(): Promise<SystemStatus> {
  const { data } = await api.get('/api/system/status');
  return data;
}

export async function getClasses(): Promise<{ classes: ClassInfo[] }> {
  const { data } = await api.get('/api/ecg/classes');
  return data;
}

export async function sampleEcgEvent(mode: string): Promise<EcgEvent> {
  const { data } = await api.get('/api/ecg/sample', { params: { mode } });
  return data;
}

export async function sendEcgEvent(event: EcgEvent): Promise<EcgSendResponse> {
  const { data } = await api.post('/api/ecg/send', event);
  return data;
}

export async function startSimulation(params: {
  mode: string;
  events_per_second: number;
  duration_seconds: number;
  n_events?: number;
}): Promise<SimulationStatus> {
  const { data } = await api.post('/api/simulation/start', params);
  return data;
}

export async function stopSimulation(): Promise<SimulationStatus> {
  const { data } = await api.post('/api/simulation/stop');
  return data;
}

export async function getSimulationStatus(): Promise<SimulationStatus> {
  const { data } = await api.get('/api/simulation/status');
  return data;
}

export async function getStatistics(): Promise<Statistics> {
  const { data } = await api.get('/api/statistics');
  return data;
}

export async function getPredictions(params: {
  limit?: number;
  predicted_class?: string;
  abnormal_only?: boolean;
}): Promise<Prediction[]> {
  const { data } = await api.get('/api/predictions', { params });
  return data;
}

export async function getExplanation(eventId: string): Promise<Explanation> {
  const { data } = await api.get(`/api/explain/${eventId}`);
  return data;
}

export async function getThreshold(): Promise<{ anomaly_threshold: number }> {
  const { data } = await api.get('/api/threshold');
  return data;
}

export async function getExperiments(): Promise<ExperimentsResponse> {
  const { data } = await api.get('/api/models/experiments');
  return data;
}

export async function getMetaheuristicsSummary(): Promise<Record<string, unknown>> {
  const { data } = await api.get('/api/models/metaheuristics');
  return data;
}

export async function getModelsStatus(): Promise<Record<string, unknown>> {
  const { data } = await api.get('/api/models/status');
  return data;
}

export interface LiveMetrics {
  total_events: number;
  latency_ms: { mean: number | null; p50: number | null; p95: number | null; p99: number | null };
  throughput_events_per_second: number | null;
  class_distribution: Record<string, number>;
  window_seconds: number;
}

export async function getLiveMetrics(windowSeconds = 60): Promise<LiveMetrics> {
  const { data } = await api.get('/api/metrics', { params: { window_seconds: windowSeconds } });
  return data;
}

export default api;
