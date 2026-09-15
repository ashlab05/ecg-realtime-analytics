# Demo Server

One command, zero infrastructure (no Kafka, no Spark, no PostgreSQL, no
Node at runtime) — the real trained models served with the built
dashboard:

```bash
# from the repo root — needs dashboard/dist/ (npm run build) to exist
python3 demo/server.py
```

Open **http://localhost:8000**. Details, prerequisites, and what differs
from the full API: see **Option 0** in `dashboard/DASHBOARD.md`.
