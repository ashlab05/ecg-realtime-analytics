# ECG Real-Time Arrhythmia Analytics

Deep Learning-Based Framework for Real-Time ECG Arrhythmia Classification in Big Data Environments.

This project implements a complete streaming analytics stack for analyzing ECG (electrocardiogram) beats from the MIT-BIH Arrhythmia Database. It features a two-stage hybrid inference pipeline (an Autoencoder anomaly gate followed by an XGBoost classifier) designed to handle highly imbalanced datasets efficiently in a real-time setting.

## Deep Learning Models & Ablation Studies

The core focus of this project is on the performance of Deep Learning models compared to traditional approaches in a high-throughput streaming environment. The system's **Production** pipeline uses a hybrid approach to balance recall and throughput, but several ablation studies were conducted to isolate the impact of the Autoencoder (AE) features and MLP baselines.

| Experiment / Ablation | Accuracy | Balanced acc. | Macro F1 | SVEB recall | VEB recall | Mean lat. (ms) | Throughput (ev/s) |
|---|---|---|---|---|---|---|---|
| **A · XGBoost only (5-class)** | 0.9919 | 0.7551 | 0.7562 | 0.940 | 0.980 | 0.0030 | 339 k |
| **B · Autoencoder only** *(DL)* | 0.9299 | 0.8585 | 0.8290 | — | — | 0.0005 | 2.66 M |
| **C · Hybrid AE → XGB (Prod)** | 0.9278 | 0.5710 | 0.5253 | 0.240 | 0.955 | 0.0018 | 569 k |
| **D · XGBoost + AE features** | 0.9914 | 0.7494 | 0.7532 | 0.938 | 0.979 | 0.0033 | 309 k |
| **E · MLP baseline** *(DL)* | 0.9383 | 0.7577 | 0.6042 | 0.964 | 0.954 | 0.0003 | 4.09 M |

**Key Findings (Ablation):**
- **Autoencoder Gate (B):** The standalone AE anomaly detector yields exceptional throughput (2.66 million events/sec) with strong balanced accuracy, effectively isolating the 96.4% majority Normal class at the cost of granular sub-class predictions.
- **Hybrid Pipeline (C):** Routing only the ~13% of beats flagged by the AE to the Stage-2 XGBoost model provides a 67% throughput increase over the XGBoost-only baseline (A), representing a practical trade-off for big data environments.
- **Feature Augmentation (D):** Appending the AE's latent representations and reconstruction errors to the XGBoost input features did not yield a significant improvement over the baseline (A).
- **MLP Baseline (E):** The Multilayer Perceptron provides the absolute lowest latency (0.0003ms) and highest throughput (4.09M ev/s), maintaining competitive SVEB and VEB recall compared to XGBoost, highlighting the viability of pure DL architectures for real-time edge deployments.

## Quick Start (Demo Mode)

The easiest way to see the system in action without setting up Kafka, Spark, or a database is to run the self-contained demo server. This runs the frontend dashboard and the API with an in-memory datastore:

```bash
# From the repository root (ensure you have built the dashboard first if developing)
python3 demo/server.py
```

Then open **http://localhost:8000** in your browser. For more details on the demo mode, see `dashboard/DASHBOARD.md`.

## Full Docker Stack

To run the full production-like streaming stack, which includes Kafka (for message brokering), Apache Spark Structured Streaming (for inference), PostgreSQL (for storing predictions), the FastAPI backend, and the React dashboard:

```bash
docker compose up --build
```

The dashboard will be available at **http://localhost:3000**. The API will be available at **http://localhost:8000**.

## Exploratory Data Analysis (EDA)

You can generate Exploratory Data Analysis figures from the MIT-BIH dataset using the provided script:
```bash
python3 scripts/generate_eda.py
```
This will output `.png` figures to `docs/figures/`.

## Documentation

For an in-depth dive into the architecture, the machine learning models (including the two-stage AE -> XGBoost pipeline), offline experiment benchmarks, and the system components, please refer to the detailed technical documentation:

- [Technical Overview & Architecture (docs/TECHNICAL.md)](docs/TECHNICAL.md)
- [Redis Streams Alternative (docs/REDIS_STREAMS.md)](docs/REDIS_STREAMS.md)
- [Dashboard Guide (dashboard/DASHBOARD.md)](dashboard/DASHBOARD.md)
- [J-Component Submission Document](docs/23mia1081_23mia1088_realtime_j_comp_submission.docx)
