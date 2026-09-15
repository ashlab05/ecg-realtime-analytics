/** WebSocket client for the live prediction feed (/ws/predictions). */
import type { Prediction } from '../types';

type PredictionHandler = (prediction: Prediction) => void;
type StateHandler = (connected: boolean) => void;

export class PredictionSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers = new Set<PredictionHandler>();
  private stateHandlers = new Set<StateHandler>();
  private reconnectTimer: number | null = null;
  private keepAliveTimer: number | null = null;
  private closedByUser = false;

  constructor(url?: string) {
    if (url) {
      this.url = url;
    } else {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const host = import.meta.env.VITE_WS_URL ?? `${window.location.host}`;
      this.url = `${proto}://${host}/ws/predictions`;
    }
  }

  connect(): void {
    this.closedByUser = false;
    this.open();
  }

  private open(): void {
    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      console.error('WebSocket construction failed', err);
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.notifyState(true);
      // Keep-alive ping every 25s (matches the server's ping/pong handling).
      this.keepAliveTimer = window.setInterval(() => {
        if (this.ws?.readyState === WebSocket.OPEN) this.ws.send('ping');
      }, 25_000);
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data as string);
        if (message.type === 'prediction' && message.data) {
          this.handlers.forEach((h) => h(message.data as Prediction));
        }
      } catch {
        // Ignore malformed frames.
      }
    };

    this.ws.onclose = () => {
      this.notifyState(false);
      this.clearTimers();
      if (!this.closedByUser) this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  private notifyState(connected: boolean): void {
    this.stateHandlers.forEach((h) => h(connected));
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer !== null) return;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.open();
    }, 3_000);
  }

  private clearTimers(): void {
    if (this.keepAliveTimer !== null) {
      clearInterval(this.keepAliveTimer);
      this.keepAliveTimer = null;
    }
  }

  onPrediction(handler: PredictionHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  onStateChange(handler: StateHandler): () => void {
    this.stateHandlers.add(handler);
    return () => this.stateHandlers.delete(handler);
  }

  close(): void {
    this.closedByUser = true;
    this.clearTimers();
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
  }
}
