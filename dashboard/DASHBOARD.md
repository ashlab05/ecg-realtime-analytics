# Dashboard Guide — CardioStream ECG Analytics

The React dashboard for the real-time ECG anomaly-detection pipeline
(`FastAPI → Kafka → Spark Structured Streaming → Autoencoder + XGBoost → PostgreSQL → WebSocket`).

It is split into **four sections** (sidebar on the left):

| Section | What it shows |
|---|---|
| **01 · Live Monitor** | System status, simulated ECG waveform (color-coded per prediction), latest prediction card with feature-contribution explanation, live statistics, anomaly timeline |
| **02 · Detect & Simulate** | The operations console — payload editor (sample → edit JSON → send → detect), dataset-replay & streaming simulator, recent detections feed |
| **03 · Model Performance** | All trained models side by side: experiments A–E (accuracy, balanced accuracy, macro/weighted F1), inference latency (mean/P95), throughput, per-class metrics, confusion matrices, anomaly threshold, live latency window |
| **04 · Analytics** | Session aggregates, class distribution, anomaly timeline, full event history table |

---

## Prerequisites

- Python 3.11+ with the project requirements installed (`pip install -r requirements.txt`)
- Node.js 18+ and npm
- Trained model artifacts in `models/` (autoencoder, Stage-2 XGBoost, preprocessing scaler).
  If missing, train them first:

  ```bash
  python3 training/train_autoencoder.py
  python3 training/train_xgboost.py --mode abnormal
  ```

  (Experiment results on the Model Performance page also need
  `python3 training/evaluate_models.py` — run once, takes a few minutes.)

## Option 0 — Demo mode (single command, nothing else running)

For demos: one Python process, **no Kafka, no Spark, no PostgreSQL, no
Node** — just the real trained models and a seeded in-memory store so
every page has content the moment it opens.

```bash
cd ecg-realtime-analytics
python3 demo/server.py        # → http://localhost:8000
```

Requires `dashboard/dist/` to exist (build once with
`npm install --include=dev && npm run build`). The server:

- serves the built dashboard at `/`,
- runs every API the dashboard calls, with `/api/ecg/send` doing the full
  validation → scaler → autoencoder gate → XGBoost path in-process,
- reports **every component as up** (there is no broker to be "down"),
- seeds 30 predictions at startup and streams live ones over the same
  `/ws/predictions` socket,
- runs the streaming simulation as an asyncio task.

Prefer iterating on the UI? `npm run dev` also works — Vite proxies
`/api` and `/ws` to this server instead of the full API.

## Option A — Local dev (recommended while iterating)

Terminal 1 — start the FastAPI backend on port **8000**:

```bash
cd ecg-realtime-analytics
PYTHONPATH=. python3 -m uvicorn api.main:app --reload --port 8000
```

Terminal 2 — start the dashboard dev server on port **5173**:

```bash
cd ecg-realtime-analytics/dashboard
npm install --include=dev     # first time only
npm run dev
```

Open **http://localhost:5173**. The Vite dev server proxies `/api`, `/ws`
and `/health` to the backend, so no CORS or URL config is needed.

> No Kafka/PostgreSQL running? Everything still works: the API falls back to
> direct in-process inference and persists to SQLite (`ecg_local.db`), and the
> status panel will show Kafka/DB as down. For the full streaming path see
> the Docker section in the root `README.md`.

## Option B — Production build served by the API

```bash
cd ecg-realtime-analytics/dashboard
npm install --include=dev
npm run build                 # outputs to dashboard/dist/
```

Then just start the API — it serves the built dashboard at
**http://localhost:8000**:

```bash
PYTHONPATH=. python3 -m uvicorn api.main:app --port 8000
```

## Option C — Docker Compose (full stack)

```bash
docker compose up --build
```

Brings up Kafka (KRaft), the API, the Spark streaming job, PostgreSQL, and
the dashboard (nginx) — dashboard on **http://localhost:8080**.

---

## Using the dashboard

### Sending payloads and detecting (Section 02)

1. **Payload Console** — pick a sample source (`random abnormal`, a specific
   class like `VEB`, …), tweak the JSON if you like (each lead has the 16
   extracted features), then hit **Send & Detect**. The result card shows the
   predicted class, confidence, reconstruction error vs threshold, latency,
   and which pipeline stage produced it.
2. **ECG Event Simulator** —
   - *Single send*: samples one heartbeat of the chosen type and publishes it.
   - *Streaming*: set events/sec, duration (and optionally a max count) and
     start a continuous simulation; watch Section 01 fill up live.

### Reading the models page (Section 03)

- The comparison table covers every experiment from
  `logs/training/experiments/` — click a row to see its per-class bars and
  confusion matrix.
- **Hybrid AE → XGBoost** (row tagged `prod`) is the deployed two-stage
  pipeline: Stage 1 autoencoder gates normal beats, Stage 2 XGBoost
  classifies the abnormal ones (SVEB/VEB/F/Q).
- The live cards at the top show the *running service's* rolling latency and
  throughput (last 120 s), separate from the offline test-batch numbers.

### Notes

- The ECG trace is a **simulated** PQRST waveform synthesized from the
  tabular features (the dataset has no raw signals) — clearly labeled as such.
- Explanations are model feature contributions, not clinical interpretations.
