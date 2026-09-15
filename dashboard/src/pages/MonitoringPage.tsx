/**
 * Detect & Simulate: the operations console. Send individual payloads,
 * run dataset-replay or continuous simulations, watch the prediction feed.
 */
import { useEffect, useState } from 'react';
import EventSimulator from '../components/EventSimulator';
import PageHeader from '../components/PageHeader';
import PayloadEditor from '../components/PayloadEditor';
import SystemStatusPanel from '../components/SystemStatusPanel';
import * as api from '../services/api';
import type { EcgEvent, Prediction, SimulationStatus } from '../types';

interface Props {
  predictions: Prediction[];
  systemStatus: ReturnType<typeof import('../hooks/useDashboardData').useSystemStatus>;
  onEventSent: (event: EcgEvent) => void;
}

export default function MonitoringPage({ predictions, systemStatus, onEventSent }: Props) {
  const [simulationStatus, setSimulationStatus] = useState<SimulationStatus>({
    running: false,
    events_sent: 0,
  });

  const refreshSimulation = () => {
    api.getSimulationStatus().then(setSimulationStatus).catch(() => undefined);
  };
  useEffect(refreshSimulation, []);

  useEffect(() => {
    if (!simulationStatus.running) return;
    const id = window.setInterval(refreshSimulation, 2_000);
    return () => window.clearInterval(id);
  }, [simulationStatus.running]);

  return (
    <main className="px-4 md:px-8 py-6 max-w-[1400px] w-full mx-auto">
      <PageHeader
        eyebrow="Section 02"
        title="Detect & Simulate"
        description="Injection console for the pipeline: craft or sample ECG event payloads, run them through anomaly detection, and drive continuous simulations at configurable rates."
      />

      <div className="space-y-6">
        <SystemStatusPanel status={systemStatus.status} />

        <div className="grid grid-cols-1 xl:grid-cols-[1.15fr_1fr] gap-6 items-start">
          <PayloadEditor />
          <div className="space-y-6">
            <EventSimulator
              onEventSent={onEventSent}
              simulationStatus={simulationStatus}
              onSimulationChange={refreshSimulation}
            />
            <RecentDetections predictions={predictions} />
          </div>
        </div>
      </div>
    </main>
  );
}

function RecentDetections({ predictions }: { predictions: Prediction[] }) {
  const recent = predictions.slice(0, 8);
  return (
    <div className="card">
      <h2 className="card-title">Recent Detections</h2>
      {recent.length === 0 ? (
        <p className="text-xs text-slate-500">No predictions yet — send an event or start a simulation.</p>
      ) : (
        <ul className="divide-y divide-surface-border/60">
          {recent.map((p) => (
            <li key={p.event_id} className="flex items-center gap-3 py-2 text-xs">
              <span
                className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                  p.is_abnormal ? 'bg-signal-abnormal' : 'bg-signal-normal'
                }`}
              />
              <span className="font-mono text-slate-400 truncate max-w-[16rem]">{p.event_id}</span>
              <span className={`ml-auto font-semibold ${p.is_abnormal ? 'text-signal-abnormal' : 'text-signal-normal'}`}>
                {p.predicted_class}
              </span>
              <span className="font-mono text-slate-500 tabular-nums w-16 text-right">
                {(p.latency_ms ?? p.inference_latency_ms ?? 0).toFixed(1)}ms
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
