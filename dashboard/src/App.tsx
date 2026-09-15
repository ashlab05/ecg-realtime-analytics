/**
 * App root: owns the cross-page state (WebSocket prediction feed, system
 * status, statistics, sent events) and switches between the four sections.
 */
import { useEffect, useState } from 'react';
import Layout, { type PageKey } from './components/Layout';
import { usePredictions, useStatistics, useSystemStatus } from './hooks/useDashboardData';
import AnalyticsPage from './pages/AnalyticsPage';
import LiveMonitorPage from './pages/LiveMonitorPage';
import ModelPerformancePage from './pages/ModelPerformancePage';
import MonitoringPage from './pages/MonitoringPage';
import type { EcgEvent } from './types';

export default function App() {
  const [page, setPage] = useState<PageKey>('monitor');
  const { predictions, connected } = usePredictions(300);
  const systemStatus = useSystemStatus(10_000);
  const stats = useStatistics(5_000);
  const [sentEvents, setSentEvents] = useState<EcgEvent[]>([]);

  const handleEventSent = (event: EcgEvent) => {
    setSentEvents((prev) => [...prev.slice(-30), event]);
  };

  useEffect(() => {
    document.title = 'CardioStream · ECG Real-Time Analytics';
  }, []);

  return (
    <Layout
      page={page}
      onNavigate={setPage}
      wsConnected={connected}
      eventsPerSecond={stats?.events_per_second ?? null}
    >
      {page === 'monitor' && (
        <LiveMonitorPage
          predictions={predictions}
          connected={connected}
          stats={stats}
          systemStatus={systemStatus}
          sentEvents={sentEvents}
        />
      )}
      {page === 'detect' && (
        <MonitoringPage predictions={predictions} systemStatus={systemStatus} onEventSent={handleEventSent} />
      )}
      {page === 'models' && <ModelPerformancePage />}
      {page === 'analytics' && <AnalyticsPage predictions={predictions} stats={stats} />}
    </Layout>
  );
}
