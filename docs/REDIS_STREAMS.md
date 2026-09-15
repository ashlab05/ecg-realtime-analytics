# Redis Streams Alternative to Kafka

> **Status: experiment.** The Kafka path (Kafka → Spark Structured Streaming)
> remains the primary architecture. This is a drop-in transport swap that was
> built to evaluate Redis Streams as a lighter-weight local alternative —
> the dashboard, REST schemas, and inference code are byte-for-byte unchanged.

## Why

Running Kafka + Spark locally is heavy (JVM ×2, health-check waits, topic
init containers) and was flaky on some machines. Redis Streams offers the
two primitives this pipeline actually needs — an append-only log and
consumer groups with acks — in a ~5 MB process.

## What changed

| Component | Kafka path | Redis path |
|---|---|---|
| Event transport | `ecg-events` topic | `ecg-events` stream (XADD) |
| Inference consumer | Spark Structured Streaming job | `streaming/redis_inference_worker.py` |
| Predictions sink | `ecg-predictions` topic | `ecg-predictions` stream |
| API's sink listener | `PredictionConsumer` thread | `RedisPredictionConsumer` thread |

The API never talks to a concrete broker. `api/services/broker.py` is a
facade selected by one env var:

```
MESSAGE_BROKER=kafka   # default
MESSAGE_BROKER=redis
```

### New files

- `api/services/redis_service.py` — `RedisEventPublisher` (XADD with
  maxlen ≈ 100 000, approximate trim) and `RedisPredictionConsumer`
  (XREADGROUP, count 50, block 1 s; XACK after the DB/WebSocket handler
  succeeds, so a crashed consumer re-reads pending entries).
- `streaming/redis_inference_worker.py` — replaces the Spark job:
  consumes `ecg-events`, runs the same two-stage inference (autoencoder
  gate → XGBoost), appends predictions to `ecg-predictions` (malformed
  events to `ecg-errors`), XACKs everything. Writes the same batch
  metrics CSV that Spark does.
- `api/services/broker.py` — the facade above (lazy imports, so the
  Redis path costs nothing when Kafka is selected).

### Configuration (`ecg_core/config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `MESSAGE_BROKER` | `kafka` | `kafka` or `redis` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `REDIS_STREAM_MAXLEN` | `100000` | per-stream cap (approx. trim) |
| `REDIS_BLOCK_MS` | `1000` | XREADGROUP block time |

### Semantics worth knowing

- **Consumer groups.** The API listens on group
  `<kafka_consumer_group>-api`, the inference worker on
  `<kafka_consumer_group>-worker`; both create the group at `$` (head) if
  missing (BUSYGROUP is swallowed). Each process uses a unique consumer
  name, so several API replicas share the predictions stream load.
- **Delivery = at-least-once.** An entry is XACKed only after its handler
  returns; poison entries (that fail even after deserialization) are
  skipped, not retried forever.
- **No rebalancing protocol.** Kafka's group coordinator is replaced by
  Redis's simpler model — fine at this scale, a real difference at large
  broker counts.

## Running it

### Docker (profile `redis`)

```bash
docker compose --profile redis up --build redis redis-worker
```

Starts Redis + the inference worker; the `api` service reads
`MESSAGE_BROKER` from the environment/.env. Note the Kafka/Spark services
still start unless excluded — to run *only* the Redis leg:

```bash
docker compose --profile redis up --build --scale kafka=0 --scale spark=0 \
    redis redis-worker api dashboard postgres
```

### Local Python

```bash
# terminal 1 — redis-server (or docker run -p 6379:6379 redis:7-alpine)
redis-server

# terminal 2 — inference worker
MESSAGE_BROKER=redis python -m streaming.redis_inference_worker

# terminal 3 — API + dashboard dev server
MESSAGE_BROKER=redis uvicorn api.main:app --reload --port 8000
```

Inspect by hand:

```bash
redis-cli XLEN ecg-events
redis-cli XLEN ecg-predictions
redis-cli XRANGE ecg-predictions - + COUNT 3
```

## What's intentionally untouched

- Dashboard (all pages, WebSocket, schemas)
- Training / models / experiments
- `streaming/spark_streaming.py` and the Kafka services
- Database layer

## Verdict so far

Code-complete and unit-reasoned, but **not yet verified against a live
Redis instance** — the Kafka path is still the one to demo. TODO: full
E2E run (API → stream → worker → predictions → dashboard) once a local
Redis is available.
