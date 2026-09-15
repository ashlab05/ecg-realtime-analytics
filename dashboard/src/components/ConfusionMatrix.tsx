/** Confusion matrix heatmap with row-normalized intensity. */
interface Props {
  labels: string[];
  matrix: number[][];
}

export default function ConfusionMatrix({ labels, matrix }: Props) {
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0) || 1);

  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-1 text-[11px] mx-auto">
        <thead>
          <tr>
            <th className="text-slate-500 font-normal text-[9.5px] uppercase tracking-wider text-right pr-1">
              true \ pred
            </th>
            {labels.map((l) => (
              <th key={l} className="text-slate-400 font-semibold px-2 pb-1 font-mono">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <td className="text-slate-400 font-semibold pr-1 text-right font-mono">{labels[i]}</td>
              {row.map((count, j) => {
                const frac = count / rowTotals[i];
                const diag = i === j;
                return (
                  <td
                    key={j}
                    title={`${labels[i]} → ${labels[j]}: ${count}`}
                    className="text-center rounded px-2 py-1.5 font-mono tabular-nums border border-surface-border/60"
                    style={{
                      background:
                        count === 0
                          ? 'rgba(255,255,255,0.02)'
                          : diag
                            ? `rgba(52, 211, 153, ${0.12 + frac * 0.75})`
                            : `rgba(248, 113, 113, ${0.12 + Math.min(frac, 0.5) * 1.1})`,
                      color: count === 0 ? '#475569' : frac > 0.45 ? '#0b1220' : '#cbd5e1',
                      fontWeight: frac > 0.2 ? 600 : 400,
                    }}
                  >
                    {count}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
