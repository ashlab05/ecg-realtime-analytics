/**
 * Analytics: aggregate statistics, class distributions, anomaly timeline,
 * and the full searchable event history.
 */
import { useMemo, useState } from 'react';
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';
import AnomalyTimeline from '../components/AnomalyTimeline';
import EventHistoryTable from '../components/EventHistoryTable';
import LiveStatistics from '../components/LiveStatistics';
import PageHeader from '../components/PageHeader';
import type { Prediction, Statistics } from '../types';

const PIE_COLORS = ['#34d399', '#22d3ee', '#818cf8', '#fbbf24', '#f87171', '#94a3b8'];

interface Props {
  predictions: Prediction[];
  stats: Statistics | null;
}

export default function AnalyticsPage({ predictions, stats }: Props) {
  const [selected, setSelected] = useState<Prediction | null>(null);

  const classDistribution = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of predictions) {
      counts.set(p.predicted_class, (counts.get(p.predicted_class) ?? 0) + 1);
    }
    return [...counts.entries()].map(([name, value]) => ({ name, value }));
  }, [predictions]);

  return (
    <main className="px-4 md:px-8 py-6 max-w-[1400px] w-full mx-auto">
      <PageHeader
        eyebrow="Section 04"
        title="Analytics"
        description="Descriptive view over the live session and offline dataset metrics. The system relies on a Deep Learning Autoencoder for anomaly detection (gating) and an XGBoost classifier for precise arrhythmia sub-classing."
      />

      <div className="space-y-6">
        <LiveStatistics stats={stats} predictions={predictions} />

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.4fr] gap-6">
          <div className="card">
            <h2 className="card-title">Live class distribution</h2>
            {classDistribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie
                    data={classDistribution}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={55}
                    outerRadius={90}
                    paddingAngle={2}
                    strokeWidth={0}
                  >
                    {classDistribution.map((entry, i) => (
                      <Cell key={entry.name} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: '#111a2e', border: '1px solid #1e2a44', borderRadius: 8, fontSize: 11 }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-xs text-slate-500">No predictions in this session yet.</p>
            )}
          </div>

          <AnomalyTimeline predictions={predictions} />
        </div>

        <div className="card mt-6 mb-6">
          <h2 className="card-title">Exploratory Data Analysis (Dataset Metrics)</h2>
          <p className="text-[11px] text-slate-400 mb-4">
            Visualizations generated from the underlying MIT-BIH dataset showing class distribution, 
            RR-interval densities (highlighting how arrhythmic beats deviate from normal rhythms), 
            and the correlation matrix of the 32 extracted features. This Deep Learning focused approach leverages these morphological and rhythmic attributes to isolate anomalies effectively.
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="rounded-lg bg-black/40 border border-surface-border p-2 flex flex-col items-center">
              <span className="text-[10px] text-slate-500 mb-2 uppercase tracking-wider">Class Distribution</span>
              <img src="/figures/class_distribution.png" alt="Class Distribution" className="w-full h-auto rounded" />
            </div>
            <div className="rounded-lg bg-black/40 border border-surface-border p-2 flex flex-col items-center">
              <span className="text-[10px] text-slate-500 mb-2 uppercase tracking-wider">RR Intervals Density</span>
              <img src="/figures/rr_intervals.png" alt="RR Intervals" className="w-full h-auto rounded" />
            </div>
            <div className="rounded-lg bg-black/40 border border-surface-border p-2 flex flex-col items-center">
              <span className="text-[10px] text-slate-500 mb-2 uppercase tracking-wider">Feature Correlation Matrix</span>
              <img src="/figures/correlation_matrix.png" alt="Correlation Matrix" className="w-full h-auto rounded" />
            </div>
          </div>
        </div>

        <EventHistoryTable
          predictions={predictions}
          onSelect={setSelected}
          selectedEventId={selected?.event_id ?? null}
        />
      </div>
    </main>
  );
}
