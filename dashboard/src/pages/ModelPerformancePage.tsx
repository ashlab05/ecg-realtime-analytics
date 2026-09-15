/**
 * Model Performance: offline ablation results (A–E), latency/throughput,
 * per-class quality, confusion matrices, plus the live inference window.
 */
import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import ConfusionMatrix from '../components/ConfusionMatrix';
import PageHeader from '../components/PageHeader';
import * as api from '../services/api';
import type { ExperimentResult } from '../types';

const CLASS_COLORS: Record<string, string> = {
  N: '#34d399',
  SVEB: '#22d3ee',
  VEB: '#818cf8',
  F: '#fbbf24',
  Q: '#f87171',
  ABNORMAL: '#f87171',
};

const fmtPct = (v: number | null | undefined) =>
  v == null ? '—' : `${(v * 100).toFixed(2)}%`;
const fmtNum = (v: number | null | undefined, digits = 2) =>
  v == null ? '—' : v.toFixed(digits);
const fmtEps = (v: number | null | undefined) =>
  v == null ? '—' : v >= 1000 ? `${(v / 1000).toFixed(0)}k/s` : `${v.toFixed(0)}/s`;

export default function ModelPerformancePage() {
  const [experiments, setExperiments] = useState<ExperimentResult[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<{ anomaly_threshold: number; threshold_percentile?: number | null } | null>(null);
  const [modelsStatus, setModelsStatus] = useState<Record<string, unknown> | null>(null);
  const [live, setLive] = useState<api.LiveMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getExperiments()
      .then((r) => {
        setExperiments(r.experiments);
        setSelected(r.experiments[0]?.name ?? null);
      })
      .catch((err) => setError(String(err)));
    api.getThreshold().then(setThreshold).catch(() => setThreshold(null));
    api.getModelsStatus().then(setModelsStatus).catch(() => setModelsStatus(null));
  }, []);

  useEffect(() => {
    const poll = () => api.getLiveMetrics(120).then(setLive).catch(() => undefined);
    poll();
    const id = window.setInterval(poll, 5_000);
    return () => window.clearInterval(id);
  }, []);

  const selectedExp = useMemo(
    () => experiments.find((e) => e.name === selected) ?? experiments[0] ?? null,
    [experiments, selected],
  );

  const comparisonData = experiments.map((e) => ({
    name: shortName(e.label),
    macro_f1: e.metrics.macro_f1 ?? 0,
    accuracy: e.metrics.accuracy ?? 0,
    balanced: e.metrics.balanced_accuracy ?? 0,
  }));

  const latencyData = experiments.map((e) => ({
    name: shortName(e.label),
    mean: e.metrics.latency_ms?.mean ?? 0,
    p95: e.metrics.latency_ms?.p95 ?? 0,
  }));

  const perClassData = useMemo(() => {
    if (!selectedExp?.metrics.per_class) return [];
    return Object.entries(selectedExp.metrics.per_class).map(([cls, m]) => ({
      cls,
      f1: m.f1,
      recall: m.recall,
      precision: m.precision,
    }));
  }, [selectedExp]);

  return (
    <main className="px-4 md:px-8 py-6 max-w-[1400px] w-full mx-auto">
      <PageHeader
        eyebrow="Section 03"
        title="Model Performance"
        description="Offline evaluation of every model variant on the held-out test split (MIT-BIH, stratified 70/15/15, seed 42), next to the live inference window of the running service."
      />

      {error && (
        <div className="card mb-6 border-signal-abnormal/40">
          <p className="text-xs text-signal-abnormal">
            Could not load experiment results: {error}. Run <code className="font-mono">python3 training/evaluate_models.py</code> to generate them.
          </p>
        </div>
      )}

      <div className="space-y-6">
        {/* Headline numbers */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            label="Production pipeline"
            value="AE → XGBoost"
            sub={selectedExp ? shortName(selectedExp.label) : '—'}
          />
          <StatCard
            label="Anomaly threshold"
            value={threshold ? threshold.anomaly_threshold.toFixed(4) : '—'}
            sub={threshold?.threshold_percentile != null ? `p${threshold.threshold_percentile} of normal val recon. error` : '95th percentile rule'}
          />
          <StatCard
            label="Live mean latency"
            value={live?.latency_ms?.mean != null ? `${live.latency_ms.mean.toFixed(2)} ms` : '—'}
            sub={live ? `${live.total_events} events in last ${live.window_seconds}s` : 'no traffic yet'}
          />
          <StatCard
            label="Live throughput"
            value={live?.throughput_events_per_second != null ? `${live.throughput_events_per_second.toFixed(1)} ev/s` : '—'}
            sub="service-side rolling window"
          />
        </div>

        {/* Comparison table */}
        <div className="card">
          <h2 className="card-title">Experiment Comparison — held-out test set</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-slate-500 border-b border-surface-border">
                  <th className="py-2 pr-4 font-medium">Experiment</th>
                  <th className="py-2 pr-4 font-medium text-right">Accuracy</th>
                  <th className="py-2 pr-4 font-medium text-right">Balanced acc</th>
                  <th className="py-2 pr-4 font-medium text-right">Macro F1</th>
                  <th className="py-2 pr-4 font-medium text-right">Weighted F1</th>
                  <th className="py-2 pr-4 font-medium text-right">Mean lat</th>
                  <th className="py-2 pr-4 font-medium text-right">P95 lat</th>
                  <th className="py-2 pr-4 font-medium text-right">Throughput</th>
                  {experiments.some((e) => e.metrics.stage2_routing_rate != null) && (
                    <th className="py-2 font-medium text-right">Stage-2 routing</th>
                  )}
                </tr>
              </thead>
              <tbody className="font-mono tabular-nums">
                {experiments.map((e) => {
                  const m = e.metrics;
                  const isProd = e.name === 'hybrid_ae_xgb';
                  return (
                    <tr
                      key={e.name}
                      onClick={() => setSelected(e.name)}
                      className={`cursor-pointer border-b border-surface-border/50 hover:bg-white/5 ${
                        selected === e.name ? 'bg-accent/5' : ''
                      }`}
                    >
                      <td className="py-2.5 pr-4 font-sans">
                        <span className={isProd ? 'text-accent-soft font-semibold' : 'text-slate-300'}>
                          {e.label}
                        </span>
                        {isProd && <span className="ml-2 badge bg-accent/15 text-accent">prod</span>}
                      </td>
                      <td className="py-2.5 pr-4 text-right text-slate-300">{fmtPct(m.accuracy)}</td>
                      <td className="py-2.5 pr-4 text-right text-slate-300">{fmtPct(m.balanced_accuracy)}</td>
                      <td className="py-2.5 pr-4 text-right font-semibold text-slate-100">{fmtPct(m.macro_f1)}</td>
                      <td className="py-2.5 pr-4 text-right text-slate-300">{fmtPct(m.weighted_f1)}</td>
                      <td className="py-2.5 pr-4 text-right text-slate-400">
                        {m.latency_ms?.mean != null ? `${fmtNum(m.latency_ms.mean, 3)} ms` : '—'}
                      </td>
                      <td className="py-2.5 pr-4 text-right text-slate-400">
                        {m.latency_ms?.p95 != null ? `${fmtNum(m.latency_ms.p95, 3)} ms` : '—'}
                      </td>
                      <td className="py-2.5 pr-4 text-right text-slate-400">{fmtEps(m.throughput_events_per_second)}</td>
                      {experiments.some((x) => x.metrics.stage2_routing_rate != null) && (
                        <td className="py-2.5 text-right text-slate-400">
                          {m.stage2_routing_rate != null ? fmtPct(m.stage2_routing_rate) : '—'}
                        </td>
                      )}
                    </tr>
                  );
                })}
                {experiments.length === 0 && (
                  <tr>
                    <td colSpan={8} className="py-6 text-center text-slate-500 font-sans">
                      No experiment results found — run the evaluation suite first.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[10.5px] text-slate-500">
            Click a row to inspect its per-class metrics and confusion matrix below. Q has 15 samples
            in the full dataset — recall on it is structurally near-zero for every model.
          </p>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <h2 className="card-title">Quality — macro F1 vs accuracy</h2>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={comparisonData} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2a44" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} interval={0} />
                <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}`} />
                <Tooltip
                  contentStyle={{ background: '#111a2e', border: '1px solid #1e2a44', borderRadius: 8, fontSize: 11 }}
                  formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar dataKey="macro_f1" name="Macro F1" fill="#22d3ee" radius={[3, 3, 0, 0]} />
                <Bar dataKey="accuracy" name="Accuracy" fill="#818cf8" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2 className="card-title">Inference latency — mean & P95 (ms, log of test batch)</h2>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={latencyData} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2a44" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} interval={0} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(v) => (v < 1 ? v.toFixed(2) : v.toFixed(1))} />
                <Tooltip
                  contentStyle={{ background: '#111a2e', border: '1px solid #1e2a44', borderRadius: 8, fontSize: 11 }}
                  formatter={(v: number) => `${v.toFixed(4)} ms`}
                />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar dataKey="mean" name="Mean" fill="#34d399" radius={[3, 3, 0, 0]} />
                <Bar dataKey="p95" name="P95" fill="#fbbf24" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Selected experiment detail */}
        {selectedExp && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <h2 className="card-title">
                Per-class metrics — <span className="text-slate-300 normal-case">{selectedExp.label}</span>
              </h2>
              {perClassData.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={perClassData} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e2a44" vertical={false} />
                    <XAxis dataKey="cls" tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 600 }} />
                    <YAxis domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ background: '#111a2e', border: '1px solid #1e2a44', borderRadius: 8, fontSize: 11 }}
                      formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                    />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Bar dataKey="precision" name="Precision" fill="#64748b" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="recall" name="Recall" radius={[3, 3, 0, 0]}>
                      {perClassData.map((d) => (
                        <Cell key={d.cls} fill={CLASS_COLORS[d.cls] ?? '#22d3ee'} />
                      ))}
                    </Bar>
                    <Bar dataKey="f1" name="F1" fill="#22d3ee" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-xs text-slate-500">No per-class breakdown recorded for this experiment.</p>
              )}
            </div>

            <div className="card">
              <h2 className="card-title">
                Confusion matrix — <span className="text-slate-300 normal-case">{selectedExp.label}</span>
              </h2>
              {selectedExp.metrics.confusion_matrix ? (
                <ConfusionMatrix
                  labels={selectedExp.metrics.confusion_matrix.labels}
                  matrix={selectedExp.metrics.confusion_matrix.matrix}
                />
              ) : (
                <p className="text-xs text-slate-500">No confusion matrix recorded.</p>
              )}
            </div>
          </div>
        )}

        {/* Pipeline + artifacts status */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <h2 className="card-title">Two-stage pipeline</h2>
            <ol className="space-y-3 text-xs text-slate-400">
              <li className="flex gap-3">
                <span className="shrink-0 font-mono text-accent">01</span>
                <span>
                  <span className="text-slate-200">Validate &amp; scale.</span> Schema check against the
                  canonical 32-feature order (16 per lead), StandardScaler transform from training artifacts.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="shrink-0 font-mono text-accent">02</span>
                <span>
                  <span className="text-slate-200">Stage 1 — autoencoder gate.</span> 32→128→64→16→64→128→32
                  reconstruction; MSE ≤ threshold ({threshold ? threshold.anomaly_threshold.toFixed(4) : '—'})
                  → classified <span className="font-mono text-signal-normal">N</span> and never touches XGBoost.
                </span>
              </li>
              <li className="flex gap-3">
                <span className="shrink-0 font-mono text-accent">03</span>
                <span>
                  <span className="text-slate-200">Stage 2 — XGBoost.</span> Only abnormal-gated beats are
                  routed to the 4-class arrhythmia classifier (SVEB / VEB / F / Q) with SHAP-style feature
                  contributions.
                </span>
              </li>
            </ol>
          </div>

          <div className="card">
            <h2 className="card-title">Loaded artifacts</h2>
            <ArtifactsStatus status={modelsStatus} />
          </div>
        </div>
      </div>
    </main>
  );
}

function shortName(label: string): string {
  return label.split('·')[0].trim();
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-surface-border bg-surface-raised p-4">
      <div className="text-[9.5px] uppercase tracking-[0.16em] text-slate-500">{label}</div>
      <div className="mt-1.5 font-mono text-lg text-slate-100 tabular-nums">{value}</div>
      {sub && <div className="mt-1 text-[10px] text-slate-500 leading-snug">{sub}</div>}
    </div>
  );
}

function ArtifactsStatus({ status }: { status: Record<string, unknown> | null }) {
  if (!status) return <p className="text-xs text-slate-500">Artifact status unavailable.</p>;
  const entries = Object.entries(status).filter(
    ([, v]) => typeof v === 'object' && v !== null && 'ready' in (v as object),
  ) as [string, { ready: boolean; [k: string]: unknown }][];
  if (entries.length === 0) {
    return (
      <pre className="text-[10.5px] font-mono text-slate-400 overflow-x-auto">
        {JSON.stringify(status, null, 2)}
      </pre>
    );
  }
  return (
    <ul className="space-y-2.5">
      {entries.map(([name, info]) => (
        <li key={name} className="flex items-center gap-3 text-xs">
          <span className={`h-1.5 w-1.5 rounded-full ${info.ready ? 'bg-signal-normal' : 'bg-signal-abnormal'}`} />
          <span className="text-slate-300">{name.replace(/_/g, ' ')}</span>
          <span className="ml-auto font-mono text-[10px] text-slate-500">
            {info.ready ? 'ready' : 'missing'}
          </span>
        </li>
      ))}
    </ul>
  );
}
