/** Consistent page header: title, one-line description, optional right slot. */
import type { ReactNode } from 'react';

interface Props {
  eyebrow: string;
  title: string;
  description?: string;
  right?: ReactNode;
}

export default function PageHeader({ eyebrow, title, description, right }: Props) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="text-[10px] uppercase tracking-[0.22em] text-accent/80 mb-1">{eyebrow}</div>
        <h1 className="text-xl font-semibold tracking-tight text-slate-100">{title}</h1>
        {description && <p className="mt-1 text-xs text-slate-500 max-w-2xl leading-relaxed">{description}</p>}
      </div>
      {right}
    </div>
  );
}
