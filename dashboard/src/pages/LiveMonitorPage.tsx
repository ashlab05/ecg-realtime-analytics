/**
 * Live Monitor: the always-on view. Simulated ECG trace, latest prediction,
 * rolling vitals and the anomaly timeline.
 */
import { useMemo, useState } from 'react';
import AnomalyTimeline from '../components/AnomalyTimeline';
import EcgMonitor from '../components/EcgMonitor';
import EventHistoryTable from '../components/EventHistoryTable';
import LiveStatistics from '../components/LiveStatistics';
import PageHeader from '../components/PageHeader';
import PredictionCard from '../components/PredictionCard';
import SystemStatusPanel from '../components/SystemStatusPanel';
import type { EcgEvent, Prediction } from '../types';

interface Props {
  predictions: Prediction[];
  connected: boolean;
  stats: ReturnType<typeof import('../hooks/useDashboardData').useStatistics>;
  systemStatus: ReturnType<typeof import('../hooks/useDashboardData').useSystemStatus>;
  sentEvents: EcgEvent[];
}

export default function LiveMonitorPage({ predictions, connected, stats, systemStatus, sentEvents }: Props) {
  const [selected, setSelected] = useState<Prediction | null>(null);

  const predictionsByEvent = useMemo(
    () => new Map(predictions.map((p) => [p.event_id, p])),
    [predictions],
  );

  const latest = predictions[0] ?? null;
  const shown = selected ?? latest;
  const currentEvent = sentEvents[sentEvents.length - 1] ?? null;

  const monitoredEvents = useMemo(() => {
    const withPreds = sentEvents.filter((e) => predictionsByEvent.has(e.event_id));
    return withPreds.slice(-8);
  }, [sentEvents, predictionsByEvent]);

  return (
    <main className="px-4 md:px-8 py-6 max-w-[1400px] w-full mx-auto">
      <PageHeader
        eyebrow="Section 01"
        title="Live Monitor"
        description="Real-time ECG event stream through the two-stage pipeline. The trace is a PQRST waveform synthesized from the event's tabular features — a visualization aid, not the raw signal."
        right={
          <span className={`badge ${connected ? 'bg-signal-normal/15 text-signal-normal' : 'bg-signal-abnormal/15 text-signal-abnormal'}`}>
            {connected ? 'WebSocket connected' : 'WebSocket disconnected'}
          </span>
        }
      />

      <div className="space-y-6">
        <SystemStatusPanel status={systemStatus.status} />

        <div className="grid grid-cols-1 xl:grid-cols-[1.6fr_1fr] gap-6">
          <EcgMonitor
            events={monitoredEvents}
            predictionsByEvent={predictionsByEvent}
            currentEventId={shown?.event_id ?? currentEvent?.event_id ?? null}
            heartbeatType={currentEvent?.true_label ?? '—'}
            streamingRate={stats?.events_per_second ?? null}
            wsConnected={connected}
          />
          <PredictionCard prediction={shown} />
        </div>

        <LiveStatistics stats={stats} predictions={predictions} />

        <AnomalyTimeline predictions={predictions} />

        <EventHistoryTable
          predictions={predictions.slice(0, 30)}
          onSelect={setSelected}
          selectedEventId={shown?.event_id ?? null}
        />
      </div>
    </main>
  );
}
