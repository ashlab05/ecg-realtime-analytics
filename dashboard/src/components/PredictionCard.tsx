/** Latest prediction result card with the "why abnormal?" explanation. */
import { useEffect, useState } from 'react';
import * as api from '../services/api';
import type { Explanation, Prediction } from '../types';
import { CLASS_LABELS } from '../types';

export default function PredictionCard({ prediction }: { prediction: Prediction | null }) {
  const [explanation, setExplanation] = useState<Explanation | null>(null);

  useEffect(() => {
    setExplanation(null);
    if (!prediction) return;
    let cancelled = false;
    api
      .getExplanation(prediction.event_id)
      .then((e) => {
        if (!cancelled) setExplanation(e);
      })
      .catch(() => {
        // Not all events are explainable (only API-sent ones are cached).
      });
    return () => {
      cancelled = true;
    };
  }, [prediction]);

  if (!prediction) {
    return (
      <div className="card flex flex-col items-center justify-center h-full min-h-64 text-slate-500">
        <p>Awaiting first prediction…</p>
        <p className="text-xs mt-1">Send an ECG event to see results here.</p>
      </div>
    );
  }

  const abnormal = prediction.is_abnormal;
  const latency = prediction.latency_ms ?? prediction.inference_latency_ms;

  return (
    <div className="card h-full">
      <div className="flex items-center justify-between mb-3">
        <h2 className="card-title mb-0">Prediction Result</h2>
        <span
          className={`badge ${abnormal ? 'bg-signal-abnormal/15 text-signal-abnormal' : 'bg-signal-normal/15 text-signal-normal'}`}
        >
          {abnormal ? 'Abnormal' : 'Normal'}
        </span>
      </div>

      <div className="flex items-baseline gap-3 mb-4">
        <span
          className={`text-4xl font-bold font-mono ${abnormal ? 'text-signal-abnormal' : 'text-signal-normal'}`}
        >
          {prediction.predicted_class}
        </span>
        <span className="text-sm text-slate-400">{CLASS_LABELS[prediction.predicted_class]}</span>
      </div>

      <dl className="grid grid-cols-2 gap-2 text-sm">
        <Row label="Confidence" value={`${(prediction.confidence * 100).toFixed(1)}%`} />
        <Row
          label="Reconstruction Error"
          value={prediction.reconstruction_error?.toFixed(4) ?? '—'}
          highlight={abnormal}
        />
        <Row label="Anomaly Threshold" value={prediction.anomaly_threshold?.toFixed(4) ?? '—'} />
        <Row label="Inference Latency" value={latency != null ? `${latency.toFixed(1)} ms` : '—'} />
        <Row label="Stage" value={prediction.stage ?? '—'} />
        <Row label="Source" value={prediction.source ?? 'kafka'} />
        <Row label="Timestamp" value={prediction.timestamp ? new Date(prediction.timestamp).toLocaleTimeString() : '—'} />
        <Row label="Event ID" value={prediction.event_id.slice(0, 8) + '…'} mono />
      </dl>

      {explanation?.top_features && explanation.top_features.length > 0 && (
        <div className="mt-4 pt-3 border-t border-surface-border">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
            Model feature contribution (SHAP)
          </h3>
          <ol className="space-y-1">
            {explanation.top_features.map((f, i) => (
              <li key={f.column} className="flex items-center gap-2 text-xs">
                <span className="text-slate-500 w-4">{i + 1}.</span>
                <span className="flex-1 truncate">{f.feature}</span>
                <span
                  className={`font-mono ${f.shap_value > 0 ? 'text-signal-abnormal' : 'text-signal-normal'}`}
                >
                  {f.shap_value > 0 ? '+' : ''}
                  {f.shap_value.toFixed(3)}
                </span>
              </li>
            ))}
          </ol>
          <p className="mt-2 text-[10px] text-slate-500">
            Model feature contribution explanation — not a clinical interpretation.
          </p>
        </div>
      )}
    </div>
  );
}

function Row({
  label,
  value,
  highlight,
  mono,
}: {
  label: string;
  value: string;
  highlight?: boolean;
  mono?: boolean;
}) {
  return (
    <div className="rounded-lg bg-surface px-3 py-2 border border-surface-border">
      <dt className="text-[11px] text-slate-500">{label}</dt>
      <dd className={`${mono ? 'font-mono' : ''} ${highlight ? 'text-signal-warn' : 'text-slate-200'}`}>
        {value}
      </dd>
    </div>
  );
}
