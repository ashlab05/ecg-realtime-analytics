/** Hooks: live prediction feed, system status polling, statistics polling. */
import { useEffect, useRef, useState } from 'react';
import { PredictionSocket } from '../services/ws';
import * as api from '../services/api';
import type { Prediction, Statistics, SystemStatus } from '../types';

/** Live prediction feed over WebSocket, with a rolling in-memory buffer. */
export function usePredictions(bufferSize = 200) {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<PredictionSocket | null>(null);

  useEffect(() => {
    const socket = new PredictionSocket();
    socketRef.current = socket;

    const offPrediction = socket.onPrediction((p) => {
      setPredictions((prev) => [p, ...prev].slice(0, bufferSize));
    });
    const offState = socket.onStateChange(setConnected);
    socket.connect();

    return () => {
      offPrediction();
      offState();
      socket.close();
    };
  }, [bufferSize]);

  return { predictions, connected };
}

/** Poll system status every `intervalMs`. */
export function useSystemStatus(intervalMs = 10_000) {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await api.getSystemStatus();
        if (!cancelled) {
          setStatus(s);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(String(err));
      }
    };
    poll();
    const id = window.setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [intervalMs]);

  return { status, error };
}

/** Poll aggregate statistics every `intervalMs`. */
export function useStatistics(intervalMs = 5_000) {
  const [stats, setStats] = useState<Statistics | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await api.getStatistics();
        if (!cancelled) setStats(s);
      } catch {
        // Backend not up yet; keep last snapshot.
      }
    };
    poll();
    const id = window.setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [intervalMs]);

  return stats;
}
