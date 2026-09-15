/**
 * Payload editor: sample a heartbeat from the dataset, tweak the raw JSON,
 * send it through the pipeline and inspect the detection result.
 */
import { useEffect, useState } from 'react';
import * as api from '../services/api';
import type { EcgEvent, Prediction } from '../types';

const SAMPLE_MODES = [
  { value: 'normal', label: 'Random normal (N)' },
  { value: 'abnormal', label: 'Random abnormal' },
  { value: 'random', label: 'Any class' },
  { value: 'N', label: 'N — Normal' },
  { value: 'SVEB', label: 'SVEB — Supraventricular ectopic' },
  { value: 'VEB', label: 'VEB — Ventricular ectopic' },
  { value: 'F', label: 'F — Fusion' },
  { value: 'Q', label: 'Q — Unknown' },
];

export default function PayloadEditor() {
  const [mode, setMode] = useState('abnormal');
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);

  const loadSample = async (m: string) => {
    setError(null);
    try {
      const ev = await api.sampleEcgEvent(m);
      setText(JSON.stringify(ev, null, 2));
    } catch (err) {
      setError(`Sampling failed: ${String(err)}`);
    }
  };

  useEffect(() => {
    loadSample(mode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const send = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const event: EcgEvent = JSON.parse(text);
      const resp = await api.sendEcgEvent(event);
      if (resp.prediction) setResult(resp.prediction);
      else setError(resp.message);
    } catch (err) {
      setError(`Send failed: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2 className="card-title">Payload Console</h2>
      <p className="text-[11px] text-slate-500 -mt-2 mb-3 leading-relaxed">
        Edit the raw event JSON (16 features × 2 leads) and send it through the full pipeline:
        validation → scaler → autoencoder gate → XGBoost when abnormal.
      </p>

      <div className="flex flex-wrap items-end gap-2 mb-3">
        <label className="text-xs text-slate-400">
          Sample from dataset
          <select
            className="input mt-1"
            value={mode}
            onChange={(e) => {
              setMode(e.target.value);
              loadSample(e.target.value);
            }}
          >
            {SAMPLE_MODES.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </label>
        <button className="btn-neutral" onClick={() => loadSample(mode)} disabled={busy}>
          Resample
        </button>
        <button className="btn-primary ml-auto" onClick={send} disabled={busy || !text.trim()}>
          {busy ? 'Detecting…' : 'Send & Detect'}
        </button>
      </div>

      <textarea
        className="input w-full h-72 font-mono text-[11px] leading-relaxed resize-y"
        spellCheck={false}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Event JSON will appear here after sampling…"
      />

      {error && (
        <div className="mt-3 rounded-lg bg-signal-abnormal/10 border border-signal-abnormal/30 px-3 py-2 text-xs text-signal-abnormal">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-3 rounded-lg border border-surface-border bg-black/30 p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              Detection result · two-stage inference pipeline
            </span>
            <span
              className={`badge ${
                result.is_abnormal
                  ? 'bg-signal-abnormal/15 text-signal-abnormal'
                  : 'bg-signal-normal/15 text-signal-normal'
              }`}
            >
              {result.predicted_class}
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <Metric label="Confidence" value={result.confidence.toFixed(3)} />
            <Metric label="Recon. error" value={result.reconstruction_error.toFixed(4)} />
            <Metric label="Threshold" value={result.anomaly_threshold.toFixed(4)} />
            <Metric
              label="Latency"
              value={`${(result.latency_ms ?? result.inference_latency_ms ?? 0).toFixed(2)} ms`}
            />
          </div>
          <div className="mt-2 text-[10px] text-slate-600 font-mono">
            stage={result.stage ?? '—'} · event={result.event_id}
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[9.5px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="font-mono text-sm text-slate-200 tabular-nums">{value}</div>
    </div>
  );
}
