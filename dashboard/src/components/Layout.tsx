/**
 * App shell: fixed sidebar navigation + per-page content.
 * Deliberately console-like: slim rail, small-caps labels, hairline
 * dividers — no card grids of gradients.
 */
import type { ReactNode } from 'react';

export type PageKey = 'monitor' | 'detect' | 'models' | 'analytics';

interface NavItem {
  key: PageKey;
  label: string;
  caption: string;
  icon: ReactNode;
}

const iconProps = {
  width: 16,
  height: 16,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
};

const PAGES: NavItem[] = [
  {
    key: 'monitor',
    label: 'Live Monitor',
    caption: 'Waveform · feed · vitals',
    icon: (
      <svg {...iconProps}>
        <path d="M2 12h4l2-7 4 14 3-9 2 2h5" />
      </svg>
    ),
  },
  {
    key: 'detect',
    label: 'Detect & Simulate',
    caption: 'Send payloads · streaming',
    icon: (
      <svg {...iconProps}>
        <circle cx="12" cy="12" r="9" />
        <circle cx="12" cy="12" r="4" />
        <path d="M12 3v2M12 19v2M3 12h2M19 12h2" />
      </svg>
    ),
  },
  {
    key: 'models',
    label: 'Model Performance',
    caption: 'Ablations A–E · latency',
    icon: (
      <svg {...iconProps}>
        <path d="M3 21h18" />
        <path d="M6 21V11M11 21V4M16 21v-8M21 21v-5" />
      </svg>
    ),
  },
  {
    key: 'analytics',
    label: 'Analytics',
    caption: 'History · distributions',
    icon: (
      <svg {...iconProps}>
        <path d="M21 12a9 9 0 1 1-9-9" />
        <path d="M21 3v6h-6" />
      </svg>
    ),
  },
];

interface LayoutProps {
  page: PageKey;
  onNavigate: (page: PageKey) => void;
  wsConnected: boolean;
  eventsPerSecond?: number | null;
  children: ReactNode;
}

export default function Layout({ page, onNavigate, wsConnected, eventsPerSecond, children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-surface text-slate-200 flex">
      {/* Sidebar */}
      <aside className="hidden md:flex w-60 shrink-0 flex-col border-r border-surface-border bg-surface-raised/40">
        <div className="px-5 pt-6 pb-5">
          <div className="flex items-center gap-2.5">
            <span className="relative flex h-2.5 w-2.5">
              <span
                className={`absolute inline-flex h-full w-full rounded-full ${
                  wsConnected ? 'animate-ping bg-signal-normal/60' : 'bg-signal-abnormal/60'
                }`}
              />
              <span
                className={`relative inline-flex h-2.5 w-2.5 rounded-full ${
                  wsConnected ? 'bg-signal-normal' : 'bg-signal-abnormal'
                }`}
              />
            </span>
            <div className="leading-tight">
              <div className="text-sm font-semibold tracking-tight text-slate-100">CardioStream</div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
                ECG Analytics
              </div>
            </div>
          </div>
        </div>

        <nav className="px-3 space-y-0.5 flex-1">
          {PAGES.map((item) => {
            const active = item.key === page;
            return (
              <button
                key={item.key}
                onClick={() => onNavigate(item.key)}
                className={`group w-full flex items-start gap-3 rounded-lg px-3 py-2.5 text-left transition-colors ${
                  active
                    ? 'bg-accent/10 text-accent-soft'
                    : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                }`}
              >
                <span className={`mt-0.5 ${active ? 'text-accent' : 'text-slate-500 group-hover:text-slate-300'}`}>
                  {item.icon}
                </span>
                <span className="leading-tight">
                  <span className="block text-[13px] font-medium">{item.label}</span>
                  <span className="block text-[10.5px] text-slate-500">{item.caption}</span>
                </span>
                {active && <span className="ml-auto mt-1 h-4 w-0.5 rounded-full bg-accent" />}
              </button>
            );
          })}
        </nav>

        <div className="px-5 py-4 border-t border-surface-border space-y-2">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-500">
            <span>Stream</span>
            <span className="font-mono text-slate-300 tabular-nums">
              {eventsPerSecond != null ? `${eventsPerSecond.toFixed(1)} ev/s` : '—'}
            </span>
          </div>
          <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-500">
            <span>Socket</span>
            <span className={wsConnected ? 'text-signal-normal' : 'text-signal-abnormal'}>
              {wsConnected ? 'connected' : 'down'}
            </span>
          </div>
          <p className="pt-1 text-[9.5px] leading-relaxed text-slate-600">
            AE → XGBoost two-stage pipeline. Simulated waveform, not raw ECG.
          </p>
        </div>
      </aside>

      {/* Content */}
      <div className="flex-1 min-w-0 flex flex-col">
        {/* Mobile nav */}
        <div className="md:hidden flex overflow-x-auto border-b border-surface-border bg-surface-raised/60">
          {PAGES.map((item) => (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              className={`px-4 py-3 text-xs font-medium whitespace-nowrap ${
                item.key === page ? 'text-accent-soft border-b-2 border-accent' : 'text-slate-400'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
        {children}
      </div>
    </div>
  );
}
