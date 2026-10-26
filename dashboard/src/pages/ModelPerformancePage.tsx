/**
 * Model Performance Page:
 * 1. Metaheuristic Autoencoder Optimization (Baseline AE, PSO-AE, GWO-AE, COA-AE).
 * 2. Full System Ablations (A–E: XGBoost, Autoencoder gate, Hybrid, Augmented, MLP).
 * 3. Publication-quality research visualizations and side-by-side metrics.
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
  'Normal (0)': '#34d399',
  'Abnormal (1)': '#f87171',
  ABNORMAL: '#f87171',
};

const fmtPct = (v: number | null | undefined) =>
  v == null ? '—' : `${(v * 100).toFixed(2)}%`;
const fmtNum = (v: number | null | undefined, digits = 4) =>
  v == null ? '—' : v.toFixed(digits);
const fmtEps = (v: number | null | undefined) =>
  v == null ? '—' : v >= 1_000_000 ? `${(v / 1_000_000).toFixed(2)}M/s` : v >= 1000 ? `${(v / 1000).toFixed(0)}k/s` : `${v.toFixed(0)}/s`;

export default function ModelPerformancePage() {
  const [activeTab, setActiveTab] = useState<'metaheuristics' | 'ablations'>('metaheuristics');
  const [experiments, setExperiments] = useState<ExperimentResult[]>([]);
  const [, setMetaheuristicsData] = useState<Record<string, any>>({});
  const [selectedMetaModel, setSelectedMetaModel] = useState<string>('gwo_ae');
  const [selectedAblation, setSelectedAblation] = useState<string>('hybrid_ae_xgb');
  const [figureCategory, setFigureCategory] = useState<'convergence' | 'roc_pr' | 'explainability' | 'robustness'>('roc_pr');
  const [threshold, setThreshold] = useState<{ anomaly_threshold: number; threshold_percentile?: number | null } | null>(null);
  const [modelsStatus, setModelsStatus] = useState<Record<string, unknown> | null>(null);
  const [live, setLive] = useState<api.LiveMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getExperiments()
      .then((r) => {
        setExperiments(r.experiments);
      })
      .catch((err) => setError(String(err)));

    api.getMetaheuristicsSummary()
      .then((data) => {
        setMetaheuristicsData(data);
      })
      .catch(() => undefined);

    api.getThreshold().then(setThreshold).catch(() => setThreshold(null));
    api.getModelsStatus().then(setModelsStatus).catch(() => setModelsStatus(null));
  }, []);

  useEffect(() => {
    const poll = () => api.getLiveMetrics(120).then(setLive).catch(() => undefined);
    poll();
    const id = window.setInterval(poll, 5_000);
    return () => window.clearInterval(id);
  }, []);

  // Filter experiments
  const metaExperiments = useMemo(() => {
    return experiments.filter((e) => ['baseline_ae', 'pso_ae', 'gwo_ae', 'coa_ae'].includes(e.name));
  }, [experiments]);

  const ablationExperiments = useMemo(() => {
    return experiments.filter((e) => !['baseline_ae', 'pso_ae', 'gwo_ae', 'coa_ae'].includes(e.name));
  }, [experiments]);

  const selectedMetaExp = useMemo(() => {
    return metaExperiments.find((e) => e.name === selectedMetaModel) ?? metaExperiments[0] ?? null;
  }, [metaExperiments, selectedMetaModel]);

  const selectedAblationExp = useMemo(() => {
    return ablationExperiments.find((e) => e.name === selectedAblation) ?? ablationExperiments[0] ?? null;
  }, [ablationExperiments, selectedAblation]);

  const metaComparisonChart = useMemo(() => {
    return metaExperiments.map((e) => {
      const extra = e.extra as any;
      const valAuprc = extra?.validation_metrics?.auprc ?? e.metrics.auprc ?? 0;
      const testAuprc = e.metrics.auprc ?? 0;
      const testAuroc = e.metrics.auroc ?? 0;
      const testF1 = e.metrics.macro_f1 ?? 0;
      const testRecall = e.metrics.recall ?? 0;
      return {
        name: e.name.replace('_ae', '').toUpperCase(),
        fullName: e.label,
        valAuprc,
        testAuprc,
        testAuroc,
        testF1,
        testRecall,
      };
    });
  }, [metaExperiments]);

  const perClassData = useMemo(() => {
    if (!selectedAblationExp?.metrics.per_class) return [];
    return Object.entries(selectedAblationExp.metrics.per_class).map(([cls, m]) => ({
      cls,
      f1: m.f1,
      recall: m.recall,
      precision: m.precision,
    }));
  }, [selectedAblationExp]);

  return (
    <main className="px-4 md:px-8 py-6 max-w-[1440px] w-full mx-auto">
      <PageHeader
        eyebrow="Section 03"
        title="Model Performance & Metaheuristic Optimization"
        description="Comprehensive evaluation comparing Autoencoder Hyperparameter Metaheuristics (Baseline, PSO, GWO, COA) alongside the production Hybrid DL pipeline on the held-out MIT-BIH test split."
      />

      {error && (
        <div className="card mb-6 border-signal-abnormal/40">
          <p className="text-xs text-signal-abnormal">
            Could not load experiment results: {error}.
          </p>
        </div>
      )}

      {/* Primary KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard
          label="Top Anomaly Gate"
          value="GWO-Tuned AE"
          sub="Test AUROC: 94.66% · AUPRC: 76.12%"
          highlight
        />
        <StatCard
          label="Anomaly Threshold"
          value={threshold ? threshold.anomaly_threshold.toFixed(4) : '0.0163'}
          sub="Calibrated p95 of Normal Val Errors"
        />
        <StatCard
          label="Throughput Champion"
          value="COA-Tuned AE"
          sub="4.92M events/sec · 0.24µs latency"
        />
        <StatCard
          label="Live Stream Status"
          value={live?.throughput_events_per_second != null ? `${live.throughput_events_per_second.toFixed(1)} ev/s` : 'Ready'}
          sub={live ? `Mean latency: ${live.latency_ms?.mean?.toFixed(2) ?? '—'} ms` : 'Standby mode'}
        />
      </div>

      {/* Navigation Tabs */}
      <div className="flex border-b border-surface-border mb-6 space-x-6 text-sm">
        <button
          onClick={() => setActiveTab('metaheuristics')}
          className={`pb-3 font-medium transition-colors border-b-2 ${
            activeTab === 'metaheuristics'
              ? 'border-accent text-accent font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Metaheuristic Autoencoder Comparison (Research Study)
        </button>
        <button
          onClick={() => setActiveTab('ablations')}
          className={`pb-3 font-medium transition-colors border-b-2 ${
            activeTab === 'ablations'
              ? 'border-accent text-accent font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          System-Wide Ablations (Ablations A–E)
        </button>
      </div>

      {activeTab === 'metaheuristics' ? (
        <div className="space-y-6">
          {/* Section: Comprehensive Comparison Table */}
          <div className="card">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <div>
                <h2 className="card-title">Autoencoder Metaheuristic Optimization Summary</h2>
                <p className="text-xs text-slate-400">
                  All models trained on Normal beats only. Outer-loop optimizers (PSO, GWO, COA) evaluated with unified candidate budget (30 evaluations each, fitness: Validation AUPRC).
                </p>
              </div>
              <span className="badge bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs self-start">
                Deterministic Seed 42 · 32 Features
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-slate-500 border-b border-surface-border">
                    <th className="py-2.5 pr-3 font-medium">Model / Optimizer</th>
                    <th className="py-2.5 pr-3 font-medium">Architecture</th>
                    <th className="py-2.5 pr-3 font-medium">LR / Decay</th>
                    <th className="py-2.5 pr-3 font-medium">Noise (σ)</th>
                    <th className="py-2.5 pr-3 font-medium text-right">Val AUPRC</th>
                    <th className="py-2.5 pr-3 font-medium text-right font-bold text-slate-300">Test AUROC</th>
                    <th className="py-2.5 pr-3 font-medium text-right font-bold text-slate-300">Test AUPRC</th>
                    <th className="py-2.5 pr-3 font-medium text-right">Test F1</th>
                    <th className="py-2.5 pr-3 font-medium text-right">Sensitivity</th>
                    <th className="py-2.5 pr-3 font-medium text-right">Specificity</th>
                    <th className="py-2.5 pr-3 font-medium text-right">MCC</th>
                    <th className="py-2.5 pr-3 font-medium text-right">Throughput</th>
                    <th className="py-2.5 font-medium text-right">Opt Time</th>
                  </tr>
                </thead>
                <tbody className="font-mono tabular-nums">
                  {metaExperiments.map((e) => {
                    const extra = (e.extra as any) ?? {};
                    const hp = extra.hyperparameters ?? {};
                    const vm = extra.validation_metrics ?? {};
                    const tm = e.metrics;
                    const isSelected = selectedMetaModel === e.name;
                    const isBest = e.name === 'gwo_ae';

                    return (
                      <tr
                        key={e.name}
                        onClick={() => setSelectedMetaModel(e.name)}
                        className={`cursor-pointer border-b border-surface-border/50 hover:bg-white/5 transition-colors ${
                          isSelected ? 'bg-accent/10 border-accent/40' : ''
                        }`}
                      >
                        <td className="py-3 pr-3 font-sans">
                          <span className={isSelected ? 'text-accent font-semibold' : 'text-slate-200'}>
                            {e.label}
                          </span>
                          {isBest && (
                            <span className="ml-2 badge bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                              Highest AUROC/PR
                            </span>
                          )}
                        </td>
                        <td className="py-3 pr-3 text-slate-400">
                          {hp.hidden_dims ? `${hp.hidden_dims.join('→')}→${hp.latent_dim}` : '—'}
                        </td>
                        <td className="py-3 pr-3 text-slate-400">
                          {hp.lr ? `${fmtNum(hp.lr, 4)} / ${fmtNum(hp.weight_decay, 1)}` : '—'}
                        </td>
                        <td className="py-3 pr-3 text-slate-400">
                          {hp.noise_std != null ? fmtNum(hp.noise_std, 3) : '0.000'}
                        </td>
                        <td className="py-3 pr-3 text-right text-slate-300">
                          {fmtPct(vm.auprc ?? extra.best_fitness)}
                        </td>
                        <td className="py-3 pr-3 text-right font-bold text-accent">
                          {fmtPct(tm.auroc)}
                        </td>
                        <td className="py-3 pr-3 text-right font-bold text-accent-soft">
                          {fmtPct(tm.auprc)}
                        </td>
                        <td className="py-3 pr-3 text-right text-slate-200 font-medium">
                          {fmtPct(tm.macro_f1 ?? tm.f1_score)}
                        </td>
                        <td className="py-3 pr-3 text-right text-emerald-400">
                          {fmtPct(tm.recall)}
                        </td>
                        <td className="py-3 pr-3 text-right text-slate-300">
                          {fmtPct(tm.specificity)}
                        </td>
                        <td className="py-3 pr-3 text-right text-slate-300">
                          {fmtNum(tm.mcc, 4)}
                        </td>
                        <td className="py-3 pr-3 text-right text-slate-400">
                          {fmtEps(tm.throughput_events_per_second)}
                        </td>
                        <td className="py-3 text-right text-slate-400">
                          {extra.optimization_time_s != null ? `${extra.optimization_time_s.toFixed(1)}s` : '0.0s'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-[11px] text-slate-500">
              Click any row to display model details, confusion matrix, threshold calibration stats, and convergence trajectory below.
            </p>
          </div>

          {/* Section: Metrics Comparison Bar Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <h2 className="card-title">AUROC & AUPRC Comparison Across Optimizers</h2>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={metaComparisonChart} margin={{ top: 12, right: 12, bottom: 4, left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2a44" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 600 }} />
                  <YAxis domain={[0.65, 1.0]} tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{ background: '#111a2e', border: '1px solid #1e2a44', borderRadius: 8, fontSize: 11 }}
                    formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="valAuprc" name="Val AUPRC (Fitness)" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="testAuroc" name="Test AUROC" fill="#34d399" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="testAuprc" name="Test AUPRC" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h2 className="card-title">Sensitivity (Recall) & F1 Score on Abnormal Beats</h2>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={metaComparisonChart} margin={{ top: 12, right: 12, bottom: 4, left: -10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2a44" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 600 }} />
                  <YAxis domain={[0.6, 0.85]} tick={{ fill: '#64748b', fontSize: 10 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{ background: '#111a2e', border: '1px solid #1e2a44', borderRadius: 8, fontSize: 11 }}
                    formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="testRecall" name="Abnormal Recall (Sensitivity)" fill="#ec4899" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="testF1" name="F1-Score" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Section: Selected Model Deep Dive */}
          {selectedMetaExp && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Card 1: Configuration & Threshold */}
              <div className="card">
                <h2 className="card-title">Hyperparameter Specs — {selectedMetaExp.name.toUpperCase()}</h2>
                {(() => {
                  const extra = (selectedMetaExp.extra as any) ?? {};
                  const hp = extra.hyperparameters ?? {};
                  return (
                    <div className="space-y-3 text-xs mt-3">
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Hidden Layers:</span>
                        <span className="font-mono text-slate-200">{hp.hidden_dims?.join(' → ') ?? '—'}</span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Latent Dimension:</span>
                        <span className="font-mono text-slate-200">{hp.latent_dim ?? '—'}</span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Learning Rate:</span>
                        <span className="font-mono text-slate-200">{fmtNum(hp.lr, 6)}</span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Weight Decay:</span>
                        <span className="font-mono text-slate-200">{hp.weight_decay ? hp.weight_decay.toExponential(2) : '0.0'}</span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Dropout Rate:</span>
                        <span className="font-mono text-slate-200">{fmtPct(hp.dropout)}</span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Denoising Noise (σ):</span>
                        <span className="font-mono text-slate-200">{fmtNum(hp.noise_std, 3)}</span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Batch Size:</span>
                        <span className="font-mono text-slate-200">{hp.batch_size ?? 256}</span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Calibrated Threshold:</span>
                        <span className="font-mono text-amber-400 font-semibold">{fmtNum(extra.threshold, 5)}</span>
                      </div>
                    </div>
                  );
                })()}
              </div>

              {/* Card 2: Confusion Matrix */}
              <div className="card">
                <h2 className="card-title">Test Set Confusion Matrix</h2>
                {selectedMetaExp.metrics.confusion_matrix ? (
                  <div className="mt-3">
                    <ConfusionMatrix
                      labels={selectedMetaExp.metrics.confusion_matrix.labels}
                      matrix={selectedMetaExp.metrics.confusion_matrix.matrix}
                    />
                    <div className="mt-3 text-[11px] text-slate-400 flex justify-between">
                      <span>FPR: {fmtPct(selectedMetaExp.metrics.false_positive_rate)}</span>
                      <span>FP / 1,000 Normal: {fmtNum(selectedMetaExp.metrics.fp_per_1000_normal, 1)}</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 mt-4">Confusion matrix not recorded.</p>
                )}
              </div>

              {/* Card 3: Timing & Latency */}
              <div className="card">
                <h2 className="card-title">Efficiency & Runtime Profile</h2>
                {(() => {
                  const extra = (selectedMetaExp.extra as any) ?? {};
                  const lat = selectedMetaExp.metrics.latency_ms;
                  return (
                    <div className="space-y-3 text-xs mt-3">
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Metaheuristic Search Time:</span>
                        <span className="font-mono text-slate-200">
                          {extra.optimization_time_s != null ? `${extra.optimization_time_s.toFixed(1)}s` : '0.0s'}
                        </span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Final Model Training Time:</span>
                        <span className="font-mono text-slate-200">
                          {extra.total_train_time_s != null ? `${extra.total_train_time_s.toFixed(1)}s` : '—'}
                        </span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Mean Inference Latency:</span>
                        <span className="font-mono text-emerald-400 font-semibold">
                          {lat?.mean != null ? `${(lat.mean * 1000).toFixed(1)} µs` : '—'}
                        </span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">P95 Inference Latency:</span>
                        <span className="font-mono text-slate-200">
                          {lat?.p95 != null ? `${(lat.p95 * 1000).toFixed(1)} µs` : '—'}
                        </span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Throughput:</span>
                        <span className="font-mono text-slate-200">
                          {fmtEps(selectedMetaExp.metrics.throughput_events_per_second)}
                        </span>
                      </div>
                      <div className="flex justify-between py-1.5 border-b border-surface-border">
                        <span className="text-slate-400">Compute Hardware:</span>
                        <span className="font-mono text-slate-200">Apple Silicon (CPU Mode)</span>
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          )}

          {/* Section: Publication Figures Gallery */}
          <div className="card">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
              <div>
                <h2 className="card-title">Publication Research Figures</h2>
                <p className="text-xs text-slate-400">
                  Vector and high-resolution plots generated directly from executed experiment artifacts in <code className="text-accent">reports/figures/</code>.
                </p>
              </div>

              {/* Subtabs for figures */}
              <div className="flex flex-wrap gap-2 text-xs">
                {(
                  [
                    ['roc_pr', 'ROC & PR Curves'],
                    ['convergence', 'Training & Convergence'],
                    ['explainability', 'Reconstruction & Features'],
                    ['robustness', 'Robustness & Drift'],
                  ] as const
                ).map(([cat, title]) => (
                  <button
                    key={cat}
                    onClick={() => setFigureCategory(cat)}
                    className={`px-3 py-1.5 rounded-lg border transition-colors ${
                      figureCategory === cat
                        ? 'bg-accent/15 border-accent text-accent font-semibold'
                        : 'border-surface-border bg-black/20 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {title}
                  </button>
                ))}
              </div>
            </div>

            {/* Display figures according to selected category */}
            {figureCategory === 'roc_pr' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="rounded-lg bg-black/40 border border-surface-border p-3 flex flex-col items-center">
                  <span className="text-[11px] text-slate-400 mb-2 font-medium">Receiver Operating Characteristic (ROC) Comparison</span>
                  <img src="/report_figures/roc_curves_comparison.png" alt="ROC Curves" className="w-full h-auto rounded" />
                </div>
                <div className="rounded-lg bg-black/40 border border-surface-border p-3 flex flex-col items-center">
                  <span className="text-[11px] text-slate-400 mb-2 font-medium">Precision-Recall (PR) Curves Comparison</span>
                  <img src="/report_figures/pr_curves_comparison.png" alt="PR Curves" className="w-full h-auto rounded" />
                </div>
              </div>
            )}

            {figureCategory === 'convergence' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="rounded-lg bg-black/40 border border-surface-border p-3 flex flex-col items-center">
                  <span className="text-[11px] text-slate-400 mb-2 font-medium">Metaheuristic Search Trajectory (Val AUPRC vs. Iteration)</span>
                  <img src="/report_figures/optimizer_convergence.png" alt="Optimizer Convergence" className="w-full h-auto rounded" />
                </div>
                <div className="rounded-lg bg-black/40 border border-surface-border p-3 flex flex-col items-center">
                  <span className="text-[11px] text-slate-400 mb-2 font-medium">Autoencoder Training & Validation Loss vs. Epoch</span>
                  <img src="/report_figures/loss_curves_comparison.png" alt="Loss Curves" className="w-full h-auto rounded" />
                </div>
              </div>
            )}

            {figureCategory === 'explainability' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="rounded-lg bg-black/40 border border-surface-border p-3 flex flex-col items-center">
                  <span className="text-[11px] text-slate-400 mb-2 font-medium">Normal vs. Abnormal Reconstruction Error Distribution</span>
                  <img src="/report_figures/reconstruction_error_distribution.png" alt="Error Distribution" className="w-full h-auto rounded" />
                </div>
                <div className="rounded-lg bg-black/40 border border-surface-border p-3 flex flex-col items-center">
                  <span className="text-[11px] text-slate-400 mb-2 font-medium">Top Contributing Features Driving Anomaly Gating</span>
                  <img src="/report_figures/top_contributing_features.png" alt="Top Features" className="w-full h-auto rounded" />
                </div>
                <div className="rounded-lg bg-black/40 border border-surface-border p-3 flex flex-col items-center">
                  <span className="text-[11px] text-slate-400 mb-2 font-medium">Representative Normal vs. Arrhythmic Feature Reconstruction Profiles</span>
                  <img src="/report_figures/representative_feature_reconstructions.png" alt="Feature Reconstructions" className="w-full h-auto rounded" />
                </div>
                <div className="rounded-lg bg-black/40 border border-surface-border p-3 flex flex-col items-center">
                  <span className="text-[11px] text-slate-400 mb-2 font-medium">Threshold Sensitivity (F1, Precision, Recall vs. Percentile)</span>
                  <img src="/report_figures/threshold_sensitivity.png" alt="Threshold Sensitivity" className="w-full h-auto rounded" />
                </div>
              </div>
            )}

            {figureCategory === 'robustness' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="rounded-lg bg-black/40 border border-surface-border p-3 flex flex-col items-center">
                  <span className="text-[11px] text-slate-400 mb-2 font-medium">Robustness to Controlled Additive Gaussian Feature Noise</span>
                  <img src="/report_figures/noise_robustness_analysis.png" alt="Noise Robustness" className="w-full h-auto rounded" />
                </div>
                <div className="rounded-lg bg-black/40 border border-surface-border p-3 flex flex-col items-center">
                  <span className="text-[11px] text-slate-400 mb-2 font-medium">Distribution Drift Simulation (Lead II Mean Shift)</span>
                  <img src="/report_figures/distribution_drift_analysis.png" alt="Distribution Drift" className="w-full h-auto rounded" />
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Section: System-Wide Ablation Experiments (Ablations A-E) */
        <div className="space-y-6">
          <div className="card">
            <h2 className="card-title">System-Wide Pipeline Ablations (A–E)</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-slate-500 border-b border-surface-border">
                    <th className="py-2.5 pr-4 font-medium">Experiment</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Accuracy</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Balanced acc</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Macro F1</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Weighted F1</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Mean Latency</th>
                    <th className="py-2.5 pr-4 font-medium text-right">Throughput</th>
                    <th className="py-2.5 font-medium text-right">Stage-2 Routing</th>
                  </tr>
                </thead>
                <tbody className="font-mono tabular-nums">
                  {ablationExperiments.map((e) => {
                    const m = e.metrics;
                    const isProd = e.name === 'hybrid_ae_xgb';
                    const isSelected = selectedAblation === e.name;
                    return (
                      <tr
                        key={e.name}
                        onClick={() => setSelectedAblation(e.name)}
                        className={`cursor-pointer border-b border-surface-border/50 hover:bg-white/5 transition-colors ${
                          isSelected ? 'bg-accent/10 border-accent/40' : ''
                        }`}
                      >
                        <td className="py-3 pr-4 font-sans">
                          <span className={isProd ? 'text-accent-soft font-semibold' : 'text-slate-300'}>
                            {e.label}
                          </span>
                          {isProd && <span className="ml-2 badge bg-accent/15 text-accent">production</span>}
                        </td>
                        <td className="py-3 pr-4 text-right text-slate-300">{fmtPct(m.accuracy)}</td>
                        <td className="py-3 pr-4 text-right text-slate-300">{fmtPct(m.balanced_accuracy)}</td>
                        <td className="py-3 pr-4 text-right font-semibold text-slate-100">{fmtPct(m.macro_f1)}</td>
                        <td className="py-3 pr-4 text-right text-slate-300">{fmtPct(m.weighted_f1)}</td>
                        <td className="py-3 pr-4 text-right text-slate-400">
                          {m.latency_ms?.mean != null ? `${fmtNum(m.latency_ms.mean, 3)} ms` : '—'}
                        </td>
                        <td className="py-3 pr-4 text-right text-slate-400">{fmtEps(m.throughput_events_per_second)}</td>
                        <td className="py-3 text-right text-slate-400">
                          {m.stage2_routing_rate != null ? fmtPct(m.stage2_routing_rate) : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {selectedAblationExp && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="card">
                <h2 className="card-title">
                  Per-class metrics — <span className="text-slate-300 normal-case">{selectedAblationExp.label}</span>
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
                  Confusion matrix — <span className="text-slate-300 normal-case">{selectedAblationExp.label}</span>
                </h2>
                {selectedAblationExp.metrics.confusion_matrix ? (
                  <ConfusionMatrix
                    labels={selectedAblationExp.metrics.confusion_matrix.labels}
                    matrix={selectedAblationExp.metrics.confusion_matrix.matrix}
                  />
                ) : (
                  <p className="text-xs text-slate-500">No confusion matrix recorded.</p>
                )}
              </div>
            </div>
          )}

          {/* Hybrid Architecture Description */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <h2 className="card-title">Production Two-Stage Architecture (Hybrid AE + XGBoost)</h2>
              <p className="text-xs text-slate-400 mb-3">
                How the Stage 1 Autoencoder gate dramatically increases pipeline efficiency by filtering out over 89% of normal beats before the heavier multi-class XGBoost classifier is invoked.
              </p>
              <ol className="space-y-3 text-xs text-slate-400">
                <li className="flex gap-3">
                  <span className="shrink-0 font-mono text-accent">01</span>
                  <span>
                    <span className="text-slate-200">Validate &amp; standardize.</span> Schema check against the 32 features, scaled using fitted StandardScaler.
                  </span>
                </li>
                <li className="flex gap-3">
                  <span className="shrink-0 font-mono text-accent">02</span>
                  <span>
                    <span className="text-slate-200">Stage 1 Autoencoder Gate.</span> MSE &le; threshold &rarr; classified Normal (N) immediately without invoking Stage 2.
                  </span>
                </li>
                <li className="flex gap-3">
                  <span className="shrink-0 font-mono text-accent">03</span>
                  <span>
                    <span className="text-slate-200">Stage 2 XGBoost Classifier.</span> Anomalies routed to 4-class classifier (SVEB / VEB / F / Q) with feature attributions.
                  </span>
                </li>
              </ol>
            </div>

            <div className="card">
              <h2 className="card-title">Loaded Model Artifacts</h2>
              <ArtifactsStatus status={modelsStatus} />
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

function StatCard({ label, value, sub, highlight }: { label: string; value: string; sub?: string; highlight?: boolean }) {
  return (
    <div className={`rounded-xl border p-4 ${
      highlight ? 'border-accent/40 bg-accent/5' : 'border-surface-border bg-surface-raised'
    }`}>
      <div className="text-[9.5px] uppercase tracking-[0.16em] text-slate-500">{label}</div>
      <div className={`mt-1.5 font-mono text-lg tabular-nums ${highlight ? 'text-accent font-semibold' : 'text-slate-100'}`}>
        {value}
      </div>
      {sub && <div className="mt-1 text-[10px] text-slate-400 leading-snug">{sub}</div>}
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
