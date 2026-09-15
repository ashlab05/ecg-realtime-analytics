/**
 * ECG Event Simulator: dataset replay + real-time streaming simulation.
 *
 * MODE A — Dataset Replay: sample a real heartbeat (normal/abnormal/
 * random/specific class) from the dataset, preview the payload, send it.
 * MODE B — Real-Time Simulation: configure rate/duration/count and
 * stream events continuously.
 */
import { useEffect, useState } from 'react';
import * as api from '../services/api';
import type { EcgEvent, SimulationStatus } from '../types';

const MODES = [
  { value: 'normal', label: 'Normal' },
  { value: 'abnormal', label: 'Abnormal' },
  { value: 'random', label: 'Random' },
  { value: 'N', label: 'N — Normal' },
  { value: 'SVEB', label: 'SVEB — Supraventricular Ectopic' },
  { value: 'VEB', label: 'VEB — Ventricular Ectopic' },
  { value: 'F', label: 'F — Fusion' },
  { value: 'Q', label: 'Q — Unknown' },
];

interface Props {
  onEventSent: (event: EcgEvent) => void;
  simulationStatus: SimulationStatus;
  onSimulationChange: () => void;
}

export default function EventSimulator({ onEventSent, simulationStatus, onSimulationChange }: Props) {
  const [mode, setMode] = useState('random');
  const [rate, setRate] = useState(2);
  const [duration, setDuration] = useState(30);
  const [nEvents, setNEvents] = useState<number | ''>('');
  const [pendingEvent, setPendingEvent] = useState<EcgEvent | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  // Sample a payload whenever the mode changes (so it can be previewed).
  useEffect(() => {
    let cancelled = false;
    api
      .sampleEcgEvent(mode)
      .then((ev) => {
        if (!cancelled) setPendingEvent(ev);
      })
      .catch(() => {
        if (!cancelled) setPendingEvent(null);
      });
    return () => {
      cancelled = true;
    };
  }, [mode]);

  const sendSingle = async () => {
    if (!pendingEvent) return;
    setBusy(true);
    setMessage(null);
    try {
      const resp = await api.sendEcgEvent(pendingEvent);
      onEventSent(pendingEvent);
      setMessage(
        resp.published_to_kafka
          ? `Event ${resp.event_id} published to Kafka — prediction incoming via WebSocket.`
          : `Kafka unavailable — direct inference used (source: ${resp.prediction?.predicted_class}).`,
      );
    } catch (err) {
      setMessage(`Send failed: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const startStream = async () => {
    setBusy(true);
    setMessage(null);
    try {
      await api.startSimulation({
        mode,
        events_per_second: rate,
        duration_seconds: duration,
        ...(nEvents !== '' && { n_events: nEvents }),
      });
      onSimulationChange();
      setMessage(`Simulation started (${mode}, ${rate} ev/s, ${duration}s).`);
    } catch (err) {
      setMessage(`Start failed: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const stopStream = async () => {
    setBusy(true);
    try {
      await api.stopSimulation();
      onSimulationChange();
      setMessage('Simulation stopped.');
    } catch (err) {
      setMessage(`Stop failed: ${String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h2 className="card-title">ECG Event Simulator</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <label className="text-xs text-slate-400">
          Heartbeat type
          <select
            className="input mt-1 w-full"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            {MODES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </label>

        <div className="grid grid-cols-3 gap-2">
          <label className="text-xs text-slate-400">
            Events/sec
            <input
              type="number"
              min={0.1}
              max={50}
              step={0.5}
              className="input mt-1 w-full"
              value={rate}
              onChange={(e) => setRate(Number(e.target.value))}
            />
          </label>
          <label className="text-xs text-slate-400">
            Duration (s)
            <input
              type="number"
              min={1}
              max={3600}
              className="input mt-1 w-full"
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
            />
          </label>
          <label className="text-xs text-slate-400">
            # events
            <input
              type="number"
              min={1}
              placeholder="∞"
              className="input mt-1 w-full"
              value={nEvents}
              onChange={(e) => setNEvents(e.target.value === '' ? '' : Number(e.target.value))}
            />
          </label>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 mt-4">
        <button className="btn-primary" onClick={sendSingle} disabled={busy || !pendingEvent}>
          Send Single ECG Event
        </button>
        <button
          className="btn-primary"
          onClick={startStream}
          disabled={busy || simulationStatus.running}
        >
          Start Streaming Simulation
        </button>
        <button className="btn-danger" onClick={stopStream} disabled={busy || !simulationStatus.running}>
          Stop Simulation
        </button>
        <button className="btn-neutral" onClick={() => setPendingEvent(null)} disabled={!pendingEvent}>
          Clear Preview
        </button>
      </div>

      {simulationStatus.running && (
        <div className="mt-3 rounded-lg bg-signal-warn/10 border border-signal-warn/30 px-3 py-2 text-xs text-signal-warn">
          Simulation running: {simulationStatus.mode} · {simulationStatus.events_sent} events sent ·{' '}
          {simulationStatus.events_per_second} ev/s
        </div>
      )}

      {message && <p className="mt-3 text-xs text-slate-400">{message}</p>}

      <details className="mt-4">
        <summary className="text-xs text-slate-400 cursor-pointer select-none">
          Outgoing payload (preview)
        </summary>
        <pre className="mt-2 max-h-56 overflow-auto rounded-lg bg-black/50 border border-surface-border p-3 text-[11px] font-mono text-slate-300">
          {pendingEvent ? JSON.stringify(pendingEvent, null, 2) : 'Sample a payload by selecting a heartbeat type.'}
        </pre>
      </details>
    </div>
  );
}
