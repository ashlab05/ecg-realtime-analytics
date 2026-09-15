/**
 * Real-time ECG monitor with a synthetic PQRST waveform.
 *
 * NOTE: The dataset contains extracted tabular features, not raw ECG
 * signals. This canvas renders a *simulated* waveform synthesized from
 * those features (see services/ecgWaveform.ts) purely for real-time
 * visualization — it is not the original raw ECG recording.
 */
import { useEffect, useRef } from 'react';
import { synthesizeBeat } from '../services/ecgWaveform';
import type { EcgEvent, Prediction } from '../types';

const FS = 360;                      // synthesis sample rate (Hz)
const SWEEP_SECONDS = 8;             // visible window
const BUFFER_SECONDS = 12;
const MAX_BUFFER = FS * BUFFER_SECONDS;

const COLOR_PENDING = '#64748b';
const COLOR_NORMAL = '#34d399';
const COLOR_ABNORMAL = '#f87171';

interface PendingBeat {
  samples: number[];
  eventId: string;
  startIdx: number;   // buffer position where this beat starts (-1 until written)
  offset: number;     // samples of this beat already written
  endIdx: number;     // exclusive end (startIdx + offset)
}

interface Props {
  /** Sent events, oldest-first is expected from the caller. */
  events: EcgEvent[];
  predictionsByEvent: Map<string, Prediction>;
  currentEventId: string | null;
  heartbeatType: string;
  streamingRate: number | null;
  wsConnected: boolean;
}

export default function EcgMonitor({
  events,
  predictionsByEvent,
  currentEventId,
  heartbeatType,
  streamingRate,
  wsConnected,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const samplesRef = useRef<Float32Array>(new Float32Array(MAX_BUFFER));
  const colorsRef = useRef<Uint8Array>(new Uint8Array(MAX_BUFFER)); // 0 pending,1 normal,2 abnormal
  const writeIdxRef = useRef(0);
  const pendingRef = useRef<PendingBeat[]>([]);
  const consumedEventsRef = useRef(new Set<string>());
  const lastFrameRef = useRef<number>(performance.now());
  const eventColorRef = useRef(new Map<string, number>()); // eventId -> color code

  // Prop-driven refs (read inside the animation loop).
  const eventsRef = useRef(events);
  const predictionsRef = useRef(predictionsByEvent);
  eventsRef.current = events;
  predictionsRef.current = predictionsByEvent;

  // Queue new events as synthesized beats.
  useEffect(() => {
    for (const ev of events) {
      if (consumedEventsRef.current.has(ev.event_id)) continue;
      consumedEventsRef.current.add(ev.event_id);
      const lead = ev.lead_ii;
      const samples = synthesizeBeat(
        {
          preRr: lead.pre_rr,
          postRr: lead.post_rr,
          pPeak: lead.p_peak,
          rPeak: lead.r_peak,
          sPeak: lead.s_peak,
          qPeak: lead.q_peak,
          tPeak: lead.t_peak,
          qrsDuration: lead.qrs_interval,
          pqInterval: lead.pq_interval,
          qtInterval: lead.qt_interval,
        },
        Math.max(0.4, Math.min(lead.post_rr, 2.0)),
      );
      pendingRef.current.push({ samples, eventId: ev.event_id, startIdx: -1, offset: 0, endIdx: -1 });
    }
    // Trim consumed set so it does not grow unbounded during long simulations.
    if (consumedEventsRef.current.size > 5_000) {
      consumedEventsRef.current = new Set([...consumedEventsRef.current].slice(-2_000));
    }
  }, [events]);

  // Recolor beats when their prediction arrives.
  useEffect(() => {
    for (const [eventId, pred] of predictionsByEvent) {
      const code = pred.is_abnormal ? 2 : 1;
      if (eventColorRef.current.get(eventId) === code) continue;
      eventColorRef.current.set(eventId, code);
      const beat = pendingRef.current.find((b) => b.eventId === eventId);
      if (beat && beat.startIdx >= 0) {
        const colors = colorsRef.current;
        for (let i = beat.startIdx; i < beat.endIdx; i++) colors[i % MAX_BUFFER] = code;
      }
    }
  }, [predictionsByEvent]);

  // Animation loop: advance the sweep, consume queued beats, render.
  useEffect(() => {
    let raf = 0;
    const render = (now: number) => {
      raf = requestAnimationFrame(render);
      const dt = Math.min((now - lastFrameRef.current) / 1000, 0.1);
      lastFrameRef.current = now;

      const samples = samplesRef.current;
      const colors = colorsRef.current;
      let toWrite = Math.round(dt * FS);

      while (toWrite > 0) {
        const beat = pendingRef.current[0];
        if (!beat) {
          // Baseline between events.
          samples[writeIdxRef.current % MAX_BUFFER] = 0;
          colors[writeIdxRef.current % MAX_BUFFER] = 0;
          writeIdxRef.current++;
          toWrite--;
          continue;
        }
        if (beat.startIdx < 0) beat.startIdx = writeIdxRef.current;
        const remaining = beat.samples.length - beat.offset;
        const n = Math.min(toWrite, remaining);
        const colorCode = eventColorRef.current.get(beat.eventId) ?? 0;
        for (let i = 0; i < n; i++) {
          samples[writeIdxRef.current % MAX_BUFFER] = beat.samples[beat.offset + i];
          colors[writeIdxRef.current % MAX_BUFFER] = colorCode;
          writeIdxRef.current++;
        }
        toWrite -= n;
        beat.offset += n;
        beat.endIdx = beat.startIdx + beat.offset;
        if (beat.offset >= beat.samples.length) {
          pendingRef.current.shift();
        }
      }

      draw();
    };

    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      // Grid.
      ctx.strokeStyle = 'rgba(148,163,184,0.08)';
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Waveform: group consecutive samples by color into path runs.
      const samples = samplesRef.current;
      const colors = colorsRef.current;
      const visible = Math.min(SWEEP_SECONDS * FS, MAX_BUFFER);
      const end = writeIdxRef.current;
      const start = Math.max(0, end - visible);
      const toX = (i: number) => ((i - start) / visible) * width;
      const toY = (v: number) => height * 0.62 - v * height * 0.34;

      const palette = [COLOR_PENDING, COLOR_NORMAL, COLOR_ABNORMAL];
      let i = start;
      while (i < end) {
        const code = colors[i % MAX_BUFFER];
        let j = i + 1;
        while (j < end && colors[j % MAX_BUFFER] === code) j++;
        ctx.strokeStyle = palette[code];
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        for (let k = i; k < j; k++) {
          const x = toX(k);
          const y = toY(samples[k % MAX_BUFFER]);
          if (k === i) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        i = j;
      }
    };

    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="card">
      <div className="flex items-start justify-between mb-2">
        <div>
          <h2 className="card-title mb-0">Real-Time ECG Monitor</h2>
          <p className="text-[11px] text-slate-500">
            Simulated ECG waveform visualization (synthesized from extracted features — not the raw
            ECG signal)
          </p>
        </div>
        <span
          className={`badge ${wsConnected ? 'bg-signal-normal/15 text-signal-normal' : 'bg-signal-abnormal/15 text-signal-abnormal'}`}
        >
          {wsConnected ? '● Live' : '● Disconnected'}
        </span>
      </div>

      <canvas ref={canvasRef} className="w-full h-52 rounded-lg bg-black/40 border border-surface-border" />

      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
        <div className="rounded-lg bg-surface px-3 py-2 border border-surface-border">
          <div className="text-slate-500">Event ID</div>
          <div className="font-mono truncate text-slate-300">{currentEventId ?? '—'}</div>
        </div>
        <div className="rounded-lg bg-surface px-3 py-2 border border-surface-border">
          <div className="text-slate-500">Heartbeat Type</div>
          <div className="font-medium">{heartbeatType}</div>
        </div>
        <div className="rounded-lg bg-surface px-3 py-2 border border-surface-border">
          <div className="text-slate-500">Streaming Rate</div>
          <div className="font-medium">{streamingRate != null ? `${streamingRate.toFixed(1)} ev/s` : '—'}</div>
        </div>
        <div className="rounded-lg bg-surface px-3 py-2 border border-surface-border">
          <div className="text-slate-500">Last Update</div>
          <div className="font-medium">{new Date().toLocaleTimeString()}</div>
        </div>
      </div>
    </div>
  );
}
