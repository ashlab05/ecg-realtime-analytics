/**
 * Synthetic ECG waveform synthesis from tabular heartbeat features.
 *
 * IMPORTANT: the dataset contains *extracted features*, not raw waveform
 * samples. This generator renders a *simulated* PQRST complex for
 * visualization purposes only — it is labeled as such in the UI and is
 * not the original raw ECG signal.
 */

export interface PqrstParams {
  preRr: number;      // seconds
  postRr: number;     // seconds
  pPeak: number;      // mV
  rPeak: number;      // mV
  sPeak: number;      // mV
  qPeak: number;      // mV
  tPeak: number;      // mV
  qrsDuration: number;  // feature units (~samples)
  pqInterval: number;
  qtInterval: number;
}

const FS = 360; // MIT-BIH sampling frequency (samples per second)

/** Gaussian bump used for P and T waves. */
function gaussian(x: number, center: number, width: number, amplitude: number): number {
  const d = (x - center) / width;
  return amplitude * Math.exp(-0.5 * d * d);
}

/**
 * Render one heartbeat (plus its post-RR rest segment) as samples.
 * Returns samples covering one full RR interval in seconds.
 */
export function synthesizeBeat(params: PqrstParams, durationSeconds: number): number[] {
  const samples: number[] = [];
  const n = Math.max(1, Math.round(durationSeconds * FS));

  // Convert feature intervals (in samples) to seconds.
  const qrsSec = params.qrsDuration / FS;
  const pqSec = Math.min(params.pqInterval / FS, 0.4);
  const qtSec = Math.min(params.qtInterval / FS, 0.6);

  const rCenter = pqSec + qrsSec / 2;
  const pCenter = Math.max(0.02, rCenter - qrsSec / 2 - pqSec * 0.55);
  const tCenter = rCenter + Math.max(0.08, qtSec - qrsSec / 2);

  for (let i = 0; i < n; i++) {
    const t = i / FS;
    let v = 0;
    // P wave
    v += gaussian(t, pCenter, 0.035, params.pPeak);
    // QRS complex: piecewise triangular Q, R, S
    const qStart = rCenter - qrsSec / 2;
    if (t >= qStart && t < rCenter) v += (params.rPeak - params.qPeak) * ((t - qStart) / (qrsSec / 2)) + params.qPeak;
    else if (t >= rCenter && t < rCenter + qrsSec / 2)
      v += (params.rPeak - params.sPeak) * (1 - (t - rCenter) / (qrsSec / 2)) + params.sPeak;
    // T wave
    v += gaussian(t, tCenter, 0.06, params.tPeak);
    // Baseline
    samples.push(v);
  }
  return samples;
}

/** Generate a scrolling waveform from a stream of feature sets. */
export function synthesizeSeries(beats: PqrstParams[], secondsPerBeat = 1.0): number[] {
  const out: number[] = [];
  for (const beat of beats) {
    out.push(...synthesizeBeat(beat, secondsPerBeat));
  }
  return out;
}
