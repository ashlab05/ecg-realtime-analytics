/** Shared TypeScript types mirroring the FastAPI schemas. */

export type EcgClass = 'N' | 'SVEB' | 'VEB' | 'F' | 'Q' | 'UNKNOWN';

export interface EcgLeadPayload {
  pre_rr: number;
  post_rr: number;
  p_peak: number;
  t_peak: number;
  r_peak: number;
  s_peak: number;
  q_peak: number;
  qrs_interval: number;
  pq_interval: number;
  qt_interval: number;
  st_interval: number;
  qrs_morph_0: number;
  qrs_morph_1: number;
  qrs_morph_2: number;
  qrs_morph_3: number;
  qrs_morph_4: number;
}

export interface EcgEvent {
  event_id: string;
  timestamp: string;
  record_id?: string | null;
  true_label?: string | null;
  lead_ii: EcgLeadPayload;
  lead_v5: EcgLeadPayload;
}

export interface Prediction {
  event_id: string;
  predicted_class: EcgClass;
  is_abnormal: boolean;
  confidence: number;
  reconstruction_error: number;
  anomaly_threshold: number;
  latency_ms?: number;
  inference_latency_ms?: number;
  timestamp?: string;
  event_timestamp?: string;
  record_id?: string | null;
  true_label?: string | null;
  stage?: string;
  stage2_probability?: number | null;
  source?: string;
  end_to_end_lag_ms?: number;
}

export interface ComponentStatus {
  status: 'up' | 'down' | 'unknown' | 'degraded';
  detail?: string | null;
}

export interface SystemStatus {
  api: ComponentStatus;
  kafka: ComponentStatus;
  spark: ComponentStatus;
  database: ComponentStatus;
  models: ComponentStatus;
  timestamp: string;
}

export interface Statistics {
  total_events: number;
  normal_events: number;
  abnormal_events: number;
  abnormal_rate: number;
  average_latency_ms: number | null;
  class_distribution: Record<string, number>;
  events_per_second: number | null;
  latency?: { mean: number | null; p50: number | null; p95: number | null; p99: number | null } | null;
}

export interface SimulationStatus {
  running: boolean;
  mode?: string | null;
  events_sent: number;
  events_per_second?: number | null;
  duration_seconds?: number | null;
  started_at?: string | null;
}

export interface ClassInfo {
  code: EcgClass;
  description: string;
  samples: number;
}

export interface ExplanationFeature {
  feature: string;
  column: string;
  shap_value: number;
}

export interface Explanation {
  event_id: string;
  predicted_class: EcgClass | null;
  is_abnormal: boolean | null;
  top_features: ExplanationFeature[] | null;
  note: string;
}

export interface EcgSendResponse {
  event_id: string;
  published_to_kafka: boolean;
  topic: string;
  message: string;
  prediction?: Prediction | null;
}

export type SimulatorMode = 'normal' | 'abnormal' | 'random' | 'N' | 'SVEB' | 'VEB' | 'F' | 'Q';

/* ---------- Offline experiment results (ablations A–E) ---------- */

export interface LatencyStats {
  mean: number | null;
  p50: number | null;
  p95: number | null;
  p99: number | null;
  max?: number | null;
}

export interface PerClassMetrics {
  precision: number;
  recall: number;
  f1: number;
}

export interface ExperimentMetrics {
  accuracy?: number | null;
  balanced_accuracy?: number | null;
  macro_f1?: number | null;
  weighted_f1?: number | null;
  f1_score?: number | null;
  auroc?: number | null;
  auprc?: number | null;
  precision?: number | null;
  recall?: number | null;
  specificity?: number | null;
  mcc?: number | null;
  false_positive_rate?: number | null;
  fp_per_1000_normal?: number | null;
  per_class?: Record<string, PerClassMetrics>;
  confusion_matrix?: { labels: string[]; matrix: number[][] };
  latency_ms?: LatencyStats;
  throughput_events_per_second?: number | null;
  stage2_routing_rate?: number | null;
  roc_auc_ovr_macro?: number | null;
}

export interface ExperimentResult {
  name: string;
  label: string;
  extra?: Record<string, unknown>;
  metrics: ExperimentMetrics;
}

export interface ExperimentsResponse {
  experiments: ExperimentResult[];
  directory: string;
}

export const CLASS_LABELS: Record<string, string> = {
  N: 'Normal',
  SVEB: 'Supraventricular Ectopic (SVEB)',
  VEB: 'Ventricular Ectopic (VEB)',
  F: 'Fusion Beat (F)',
  Q: 'Unknown Beat (Q)',
  UNKNOWN: 'Unknown',
};
