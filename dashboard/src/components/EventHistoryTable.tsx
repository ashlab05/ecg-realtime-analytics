/** Event history table with filtering. */
import { useMemo, useState } from 'react';
import type { Prediction } from '../types';

const FILTERS = ['All', 'Normal', 'Abnormal'] as const;

interface Props {
  predictions: Prediction[];
  onSelect: (prediction: Prediction) => void;
  selectedEventId: string | null;
}

export default function EventHistoryTable({ predictions, onSelect, selectedEventId }: Props) {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>('All');
  const [classFilter, setClassFilter] = useState('All');

  const classes = useMemo(
    () => ['All', ...Array.from(new Set(predictions.map((p) => p.predicted_class)))],
    [predictions],
  );

  const rows = useMemo(() => {
    return predictions.filter((p) => {
      if (filter === 'Normal' && p.is_abnormal) return false;
      if (filter === 'Abnormal' && !p.is_abnormal) return false;
      if (classFilter !== 'All' && p.predicted_class !== classFilter) return false;
      return true;
    });
  }, [predictions, filter, classFilter]);

  return (
    <div className="card">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <h2 className="card-title mb-0">Event History</h2>
        <div className="flex gap-2 items-center">
          <div className="flex rounded-lg overflow-hidden border border-surface-border">
            {FILTERS.map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-2.5 py-1 text-xs ${
                  filter === f ? 'bg-accent/20 text-accent-soft' : 'text-slate-400 hover:bg-white/5'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
          <select
            className="input text-xs"
            value={classFilter}
            onChange={(e) => setClassFilter(e.target.value)}
          >
            {classes.map((c) => (
              <option key={c} value={c}>
                {c === 'All' ? 'All classes' : c}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="overflow-x-auto max-h-96 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-surface-raised">
            <tr className="text-left text-slate-500 border-b border-surface-border">
              <th className="px-3 py-2 font-medium">Timestamp</th>
              <th className="px-3 py-2 font-medium">Event ID</th>
              <th className="px-3 py-2 font-medium">Input Type</th>
              <th className="px-3 py-2 font-medium">Predicted</th>
              <th className="px-3 py-2 font-medium">Confidence</th>
              <th className="px-3 py-2 font-medium">Anomaly Score</th>
              <th className="px-3 py-2 font-medium">Latency</th>
              <th className="px-3 py-2 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={8} className="px-3 py-6 text-center text-slate-600">
                  No events yet.
                </td>
              </tr>
            )}
            {rows.map((p) => (
              <tr
                key={p.event_id}
                onClick={() => onSelect(p)}
                className={`border-b border-surface-border/50 cursor-pointer hover:bg-white/5 ${
                  selectedEventId === p.event_id ? 'bg-accent/10' : ''
                }`}
              >
                <td className="px-3 py-1.5 text-slate-400">
                  {p.timestamp ? new Date(p.timestamp).toLocaleTimeString() : '—'}
                </td>
                <td className="px-3 py-1.5 font-mono text-slate-500">{p.event_id.slice(0, 8)}…</td>
                <td className="px-3 py-1.5">{p.true_label ?? '—'}</td>
                <td className="px-3 py-1.5 font-semibold">{p.predicted_class}</td>
                <td className="px-3 py-1.5">{(p.confidence * 100).toFixed(1)}%</td>
                <td className="px-3 py-1.5 font-mono">{p.reconstruction_error?.toFixed(4) ?? '—'}</td>
                <td className="px-3 py-1.5">
                  {(p.latency_ms ?? p.inference_latency_ms)?.toFixed(1) ?? '—'} ms
                </td>
                <td className="px-3 py-1.5">
                  <span
                    className={`badge ${
                      p.is_abnormal
                        ? 'bg-signal-abnormal/15 text-signal-abnormal'
                        : 'bg-signal-normal/15 text-signal-normal'
                    }`}
                  >
                    {p.is_abnormal ? 'Abnormal' : 'Normal'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
