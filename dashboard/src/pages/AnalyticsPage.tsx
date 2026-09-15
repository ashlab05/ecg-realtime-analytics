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
        description="Descriptive view over the live session: aggregate statistics, class mix of the prediction feed, anomaly occurrences over time, and the full event history."
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

        <EventHistoryTable
          predictions={predictions}
          onSelect={setSelected}
          selectedEventId={selected?.event_id ?? null}
        />
      </div>
    </main>
  );
}
