/** Timeline of reconstruction errors vs the anomaly threshold. */
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { Prediction } from '../types';

export default function AnomalyTimeline({ predictions }: { predictions: Prediction[] }) {
  const data = [...predictions]
    .reverse()
    .slice(-80)
    .map((p) => ({
      time: new Date(p.timestamp ?? p.event_timestamp ?? Date.now()).toLocaleTimeString(),
      error: p.reconstruction_error ?? 0,
      threshold: p.anomaly_threshold ?? 0,
    }));

  const threshold = data.length > 0 ? data[data.length - 1].threshold : 0;

  return (
    <div className="card">
      <h2 className="card-title">Anomaly Timeline — Reconstruction Error vs Threshold</h2>
      {data.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-xs text-slate-600">
          No anomaly scores yet.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="errFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(148,163,184,0.1)" />
            <XAxis dataKey="time" stroke="#64748b" fontSize={10} />
            <YAxis stroke="#64748b" fontSize={11} />
            <Tooltip
              contentStyle={{ background: '#111a2e', border: '1px solid #1e2a44', borderRadius: 8 }}
            />
            <Area type="monotone" dataKey="error" stroke="#22d3ee" fill="url(#errFill)" strokeWidth={1.5} />
            {threshold > 0 && (
              <ReferenceLine
                y={threshold}
                stroke="#f87171"
                strokeDasharray="4 4"
                label={{ value: 'threshold', fill: '#f87171', fontSize: 10, position: 'insideTopRight' }}
              />
            )}
            <Line type="monotone" dataKey="threshold" stroke="transparent" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
