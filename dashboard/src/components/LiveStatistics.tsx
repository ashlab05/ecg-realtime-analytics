/** Live statistics: totals, latency, rate, abnormal share — with charts. */
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { Prediction, Statistics } from '../types';

const CLASS_COLORS: Record<string, string> = {
  N: '#34d399',
  SVEB: '#fbbf24',
  VEB: '#f87171',
  F: '#a78bfa',
  Q: '#f472b6',
  UNKNOWN: '#94a3b8',
};

interface Props {
  stats: Statistics | null;
  predictions: Prediction[];
}

export default function LiveStatistics({ stats, predictions }: Props) {
  const totals = {
    total: stats?.total_events ?? 0,
    normal: stats?.normal_events ?? 0,
    abnormal: stats?.abnormal_events ?? 0,
    rate: stats?.events_per_second ?? 0,
    latency: stats?.latency?.p50 ?? stats?.average_latency_ms ?? null,
    abnormalRate: stats ? (stats.abnormal_rate * 100).toFixed(1) : '0.0',
  };

  const distribution = Object.entries(stats?.class_distribution ?? {}).map(([cls, count]) => ({
    name: cls,
    value: count,
  }));

  // Anomaly timeline: predictions ordered oldest -> newest, latency + error.
  const timeline = [...predictions]
    .reverse()
    .slice(-60)
    .map((p) => ({
      time: new Date(p.timestamp ?? p.event_timestamp ?? Date.now()).toLocaleTimeString(),
      latency: p.latency_ms ?? p.inference_latency_ms ?? 0,
      error: p.reconstruction_error ?? 0,
      threshold: p.anomaly_threshold ?? 0,
    }));

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      {/* Stat tiles */}
      <div className="card xl:col-span-3">
        <h2 className="card-title">Live Statistics</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          <Tile label="Total Events" value={totals.total.toLocaleString()} />
          <Tile label="Normal" value={totals.normal.toLocaleString()} tone="normal" />
          <Tile label="Abnormal" value={totals.abnormal.toLocaleString()} tone="abnormal" />
          <Tile label="Avg Latency" value={totals.latency != null ? `${totals.latency.toFixed(1)} ms` : '—'} />
          <Tile label="Events / sec" value={totals.rate.toFixed(1)} />
          <Tile label="Abnormal Rate" value={`${totals.abnormalRate}%`} tone="warn" />
        </div>
      </div>

      {/* Predicted class distribution (pie) */}
      <div className="card">
        <h2 className="card-title">Predicted Class Distribution</h2>
        {distribution.length === 0 ? (
          <EmptyChart />
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={distribution} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80} paddingAngle={2}>
                {distribution.map((d) => (
                  <Cell key={d.name} fill={CLASS_COLORS[d.name] ?? '#94a3b8'} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: '#111a2e', border: '1px solid #1e2a44', borderRadius: 8 }}
              />
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Class frequency bar chart */}
      <div className="card">
        <h2 className="card-title">Class Frequencies</h2>
        {distribution.length === 0 ? (
          <EmptyChart />
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={distribution}>
              <CartesianGrid stroke="rgba(148,163,184,0.1)" />
              <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
              <Tooltip
                contentStyle={{ background: '#111a2e', border: '1px solid #1e2a44', borderRadius: 8 }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {distribution.map((d) => (
                  <Cell key={d.name} fill={CLASS_COLORS[d.name] ?? '#94a3b8'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Latency timeline */}
      <div className="card">
        <h2 className="card-title">Inference Latency Timeline</h2>
        {timeline.length === 0 ? (
          <EmptyChart />
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={timeline}>
              <CartesianGrid stroke="rgba(148,163,184,0.1)" />
              <XAxis dataKey="time" stroke="#64748b" fontSize={10} />
              <YAxis stroke="#64748b" fontSize={11} unit="ms" />
              <Tooltip
                contentStyle={{ background: '#111a2e', border: '1px solid #1e2a44', borderRadius: 8 }}
              />
              <Line type="monotone" dataKey="latency" stroke="#22d3ee" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

function Tile({ label, value, tone }: { label: string; value: string; tone?: 'normal' | 'abnormal' | 'warn' }) {
  const toneClass =
    tone === 'normal' ? 'text-signal-normal' : tone === 'abnormal' ? 'text-signal-abnormal' : tone === 'warn' ? 'text-signal-warn' : 'text-slate-100';
  return (
    <div className="rounded-lg bg-surface px-3 py-3 border border-surface-border text-center">
      <div className={`text-xl font-bold ${toneClass}`}>{value}</div>
      <div className="text-[11px] text-slate-500 mt-1">{label}</div>
    </div>
  );
}

function EmptyChart() {
  return (
    <div className="h-[220px] flex items-center justify-center text-xs text-slate-600">
      No data yet — send ECG events to populate charts.
    </div>
  );
}
