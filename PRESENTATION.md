# Final Presentation: ECG Real-Time Arrhythmia Analytics

Welcome to the presentation guide for the **Deep Learning-Based Framework for Real-Time ECG Arrhythmia Classification**. This document contains everything you need to showcase the project, from running the demo to explaining the core architectural decisions.

---

## 1. Project Idea & Reasoning

**The Problem:**
ECG datasets are notoriously imbalanced. In the MIT-BIH Arrhythmia Database, approximately **96.4%** of the heartbeats are normal. Traditional models often struggle to maintain high throughput in real-time scenarios because they must evaluate complex features for every single beat—even the vast majority that are perfectly fine.

**Our Solution (The Hybrid Architecture):**
We implemented a two-stage hybrid pipeline to explicitly solve this big-data streaming problem:
1. **Deep Learning Autoencoder (Stage 1 / Anomaly Gate):** 
   Trained exclusively on *normal* beats. It acts as an extremely fast, high-throughput anomaly detector. If a heartbeat can be accurately reconstructed by the Autoencoder, it's flagged as normal (the 96.4% majority) and bypasses the rest of the pipeline.
2. **XGBoost Classifier (Stage 2):** 
   Only the beats that the Autoencoder fails to reconstruct (the anomalous minority) are passed to the XGBoost classifier. This model performs precise multi-class classification (SVEB, VEB, F, Q) on these complex cases.

**Why this matters:**
This approach leverages the strengths of both paradigms. The Deep Learning model provides a massive throughput win (gating out the normal majority with extremely low latency), while the XGBoost model provides explainable, high-precision classification for the abnormal cases.

---

## 2. Running & Closing the Application

For a presentation setting, the **Demo Mode** is the safest and most reliable way to showcase the system. It collapses the entire architecture (FastAPI, React Dashboard, WebSocket broadcaster, and Model Inference) into a single dependency-free Python process.

### How to Run (Demo Mode)

1. Open a terminal and navigate to the project root:
   ```bash
   cd /Users/mohammedashlab/Developer/realtime/ecg-realtime-analytics
   ```
2. Start the self-contained demo server:
   ```bash
   python3 demo/server.py
   ```
3. Open your browser and navigate to: **[http://localhost:8000](http://localhost:8000)**

### How to Close
When you are finished with the presentation, simply go back to the terminal where the server is running and press `Ctrl+C`. This safely shuts down the server.

*(Note: If you have Docker running and want to show the full distributed architecture, you can use `docker compose up --build`. Close it with `docker compose down`.)*

---

## 3. Dashboard Walkthrough

During your presentation, guide your audience through the dashboard sections as follows:

### Section 01: Detect & Simulate
- **What to show:** This is the control center. Start a "Random" simulation at about 10-20 events per second. 
- **Talking points:** Explain that the system is generating realistic ECG tabular features and streaming them into the FastAPI backend just like a real Kafka telemetry stream.

### Section 02: Live Monitor
- **What to show:** The real-time synthetic ECG waveform and the live prediction feed.
- **Talking points:** Point out the latency counter (usually sub-millisecond). Show how abnormal beats (highlighted in red/yellow) trigger the XGBoost explanations (SHAP contributions), while normal beats are efficiently gated out by the Autoencoder.

### Section 03: Model Performance
- **What to show:** The comparison table and the "Hybrid Architecture" breakdown.
- **Talking points:** This is where you prove your project's thesis. Highlight the **Production (AE → XGB)** row compared to the baselines. Show how the Deep Learning baseline (Autoencoder only) achieves maximum throughput, and how the hybrid approach balances that throughput with XGBoost's recall. 

### Section 04: Analytics & EDA
- **What to show:** The Live Class Distribution pie chart, the Anomaly Timeline, and the newly added **Exploratory Data Analysis** section.
- **Talking points:** Walk through the dataset metrics. Show the Class Distribution (highlighting the massive imbalance), the RR-Intervals density (showing how arrhythmias deviate rhythmically), and the Correlation Matrix. Explain how these Deep Learning features allow the Autoencoder to so accurately isolate anomalies.
