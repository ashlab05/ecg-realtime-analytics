/** System status panel: Kafka / Spark / API / DB / models. */
import type { SystemStatus } from '../types';

const LABELS: Record<string, string> = {
  api: 'API',
  kafka: 'Kafka',
  spark: 'Spark Streaming',
  database: 'PostgreSQL',
  models: 'Models',
};

const DOT: Record<string, string> = {
  up: 'bg-signal-normal',
  down: 'bg-signal-abnormal',
  degraded: 'bg-signal-warn',
  unknown: 'bg-slate-500',
};

export default function SystemStatusPanel({ status }: { status: SystemStatus | null }) {
  return (
    <div className="card">
      <h2 className="card-title">System Status</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
        {(Object.keys(LABELS) as Array<keyof typeof LABELS>).map((key) => {
          const s = status?.[key as keyof Omit<SystemStatus, 'timestamp'>];
          const state = s?.status ?? 'unknown';
          return (
            <div
              key={key}
              className="flex items-center gap-2 rounded-lg bg-surface px-3 py-2 border border-surface-border"
              title={s?.detail ?? undefined}
            >
              <span className={`h-2.5 w-2.5 rounded-full ${DOT[state]} animate-pulse`} />
              <div className="min-w-0">
                <div className="text-xs text-slate-400">{LABELS[key]}</div>
                <div className="text-sm font-medium capitalize truncate">{state}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
