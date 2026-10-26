# Final Experiment Report: Deep Learning Autoencoder Optimization for Real-Time ECG Anomaly Detection

**Author:** Research & Engineering Team  
**Date:** 2026-10-24  
**Project:** ECG Real-Time Arrhythmia Analytics  
**Repository:** `realtime/ecg-realtime-analytics`  
**Evaluation Dataset:** MIT-BIH Arrhythmia Feature Database ($N=100,689$ beats)  

---

## 1. Executive Summary

This research report presents the complete empirical investigation comparing metaheuristic hyperparameter optimization techniques for deep autoencoder anomaly detection in electrocardiogram (ECG) telemetry.

In real-time healthcare monitoring, severe class imbalance (where normal sinus rhythm represents over 89% of all monitored beats) poses computational and diagnostic challenges. To address this, the project utilizes a **Two-Stage Hybrid Architecture**:
1. **Stage 1 (Deep Learning Autoencoder Gate):** Trained exclusively on normal heartbeats. Reconstruction error serves as an anomaly score. Beats below a calibrated threshold are immediately classified as normal ($N$), bypassing downstream classifiers and conserving computational budget.
2. **Stage 2 (Supervised Arrhythmia Classifier):** Multi-class XGBoost classifier assigning precise diagnostic labels (`SVEB`, `VEB`, `F`, `Q`) solely to beats flagged as anomalous by Stage 1.

The central research objective addressed in this report is the **fair, systematic comparison of outer-loop metaheuristic optimizers**—Particle Swarm Optimization (**PSO**), Grey Wolf Optimizer (**GWO**), and Coati Optimization Algorithm (**COA**)—against an un-tuned baseline autoencoder trained with Adam gradient descent.

### Key Empirical Findings:
1. **GWO-Tuned Autoencoder is the Top Diagnostic Performer:** GWO identified a 32-dimensional bottleneck with denoising regularization ($\sigma = 0.1157$), achieving the highest Test AUROC (**0.9466**) and Test AUPRC (**0.7612**), with top abnormal detection sensitivity of **78.50%** (detecting 1,249 of 1,591 test anomalies at a low false-positive rate of 5.48%).
2. **COA-Tuned Autoencoder is the High-Throughput Edge Champion:** COA converged to an ultra-compact architecture (`32 → 64 → 32 → 12`, 9,280 parameters), achieving **4.92 million heartbeats/second** throughput with a latency of **0.240 microseconds per beat**, ideal for resource-constrained wearable Holter devices.
3. **PSO-Tuned Autoencoder Provides Rapid Convergence:** PSO uncovered effective weight decay and dropout settings that triggered early stopping at epoch 27 (7.73 s training time), delivering **0.9451 Test AUROC**.

---

## 2. Experimental Setup & Protocol

### 2.1 Dataset & Feature Representation
- **Source:** MIT-BIH Arrhythmia Database (`data/raw/MIT-BIH Arrhythmia Database.csv`).
- **Sample Count:** 100,689 beats from 44 unique ambulatory patient recordings.
- **Feature Space:** 32 pre-extracted numerical features per heartbeat (16 Lead II, 16 Lead V). The input is a standardized tabular feature vector reflecting cardiac intervals (pre-RR, post-RR, QRS duration, PQ, QT, ST intervals) and morphological wave amplitudes/segments (`pPeak`, `rPeak`, `tPeak`, `qrs_morph0` to `qrs_morph4`), rather than raw unprocessed time series.
- **Class Breakdown:**
  - Normal ($N$): 90,083 beats (89.47%) $\rightarrow$ `binary_label = 0` (Negative)
  - Ventricular Ectopic Beat ($VEB$): 7,009 beats (6.96%) $\rightarrow$ `binary_label = 1` (Positive)
  - Supraventricular Ectopic Beat ($SVEB$): 2,779 beats (2.76%) $\rightarrow$ `binary_label = 1` (Positive)
  - Fusion Beat ($F$): 803 beats (0.80%) $\rightarrow$ `binary_label = 1` (Positive)
  - Unknown Beat ($Q$): 15 beats (0.015%) $\rightarrow$ `binary_label = 1` (Positive)
  - Total Anomalies: 10,606 beats (10.53%). Class imbalance ratio: 8.5 : 1.

### 2.2 Preprocessing & Data Leakage Controls
- **Partitioning:** Stratified 70/15/15 split using random seed `42`:
  - **Training Split:** 70,482 samples (63,057 Normal, 7,425 Abnormal)
  - **Validation Split:** 15,103 samples (13,513 Normal, 1,590 Abnormal)
  - **Held-Out Test Split:** 15,104 samples (13,513 Normal, 1,591 Abnormal)
- **Leakage Prevention:** `StandardScaler` parameters were fit strictly on the training partition (`X[idx_train]`).
- **Semi-Supervised Protocol:** Autoencoder weights were trained **strictly on Normal training heartbeats**. The labeled validation set was utilized solely to evaluate candidate hyperparameter configurations via validation AUPRC. The held-out test split was kept untouched until final evaluation.

### 2.3 Optimization Search Space & Budget
The continuous search space $[0, 1]^7$ mapped to 7 key hyperparameters:
- Latent dimension $\in \{8, 12, 16, 24, 32\}$
- Hidden layer width $\in \{64, 128, 192, 256\}$ (structured as `(width, max(32, width // 2))`)
- Learning rate $\in [10^{-4}, 10^{-2}]$ (log-scale)
- Weight decay $\in [10^{-6}, 10^{-2}]$ (log-scale)
- Dropout $\in [0.0, 0.30]$
- Denoising input noise $\sigma \in [0.0, 0.25]$ (Gaussian noise added to training inputs)
- Batch size $\in \{128, 256, 512\}$

**Unified Evaluation Budget:**
- Population / Pack Size: $N = 6$
- Iterations: $T = 5$
- Candidate Evaluations per Optimizer: 36 evaluations (6 initial + 30 iterative)
- Candidate Training Limit: 15 epochs with early stopping patience = 3
- Fitness Objective: Validation AUPRC (Average Precision)

### 2.4 Threshold Calibration Policy
The anomaly decision threshold was calibrated on normal validation reconstruction errors:
$$\theta = \text{Percentile}_{95}(\{ \text{MSE}(x_i, \hat{x}_i) \mid x_i \in \mathcal{D}_{val}^{Normal} \})$$
Reconstruction errors exceeding $\theta$ trigger an abnormal decision (`pred = 1`).

---

## 3. Comprehensive Experimental Results

### 3.1 Primary Results Matrix (Held-Out Test Split)

| Metric | Baseline AE + Adam | PSO-Tuned AE + Adam | GWO-Tuned AE + Adam | COA-Tuned AE + Adam |
| :--- | :---: | :---: | :---: | :---: |
| **Selected Architecture** | `128 → 64 → 16` | `192 → 96 → 24` | `192 → 96 → 32` | `64 → 32 → 12` |
| **Model Type** | Standard AE | Denoising AE | Denoising AE | Denoising AE |
| **Denoising Noise ($\sigma$)** | 0.000 | 0.181 | 0.116 | 0.112 |
| **Learning Rate** | `0.00100` | `0.00416` | `0.00126` | `0.00138` |
| **Weight Decay** | `0.0` | `2.53e-5` | `2.20e-6` | `4.67e-5` |
| **Batch Size** | 256 | 512 | 512 | 128 |
| **Threshold ($\theta$)** | `0.05783` | `0.05357` | `0.01631` | `0.06481` |
| **Val AUPRC (Fitness)** | `0.7505` | `0.7385` | **0.7726** | `0.7098` |
| **Val AUROC** | `0.9430` | **0.9492** | `0.9483` | `0.9235` |
| **Test AUROC** | `0.9404` | `0.9451` | **0.9466** | `0.9216` |
| **Test AUPRC** | `0.7346` | `0.7289` | **0.7612** | `0.6966` |
| **Test F1-Score** | `0.6973` | `0.6903` | **0.6976** | `0.6669` |
| **Test Precision** | **0.6394** | `0.6294` | `0.6276` | `0.6126` |
| **Test Sensitivity (Recall)** | `0.7668` | `0.7643` | **0.7850** | `0.7316` |
| **Test Specificity** | **0.9491** | `0.9470` | `0.9452` | `0.9455` |
| **Test Accuracy** | **0.9299** | `0.9278` | `0.9283` | `0.9230` |
| **Test Balanced Accuracy** | `0.8580` | `0.8557` | **0.8651** | `0.8386` |
| **Test MCC** | `0.6615` | `0.6538` | **0.6628** | `0.6269` |
| **False Positive Rate (FPR)** | **0.0509** | `0.0530` | `0.0548` | `0.0545` |
| **FP / 1,000 Normal Beats** | **50.9** | `53.0` | `54.8` | `54.5` |
| **Inference Latency (Mean)** | `0.357 µs` | `0.573 µs` | `0.551 µs` | **0.240 µs** |
| **Inference Throughput** | `2.97M ev/s` | `1.92M ev/s` | `1.90M ev/s` | **4.92M ev/s** |
| **Optimization Time** | `0.0 s` | `108.7 s` | `139.6 s` | `118.0 s` |
| **Training Duration** | `11.8 s` (50 ep) | **7.7 s** (27 ep) | `14.3 s` (50 ep) | `12.4 s` (50 ep) |

### 3.2 Test Confusion Matrix Counts ($N=15,104$)
- **Baseline AE:** $TN = 12,825$, $FP = 688$, $FN = 371$, $TP = 1,220$
- **PSO-Tuned AE:** $TN = 12,797$, $FP = 716$, $FN = 375$, $TP = 1,216$
- **GWO-Tuned AE:** $TN = 12,772$, $FP = 741$, $FN = 342$, $TP = 1,249$
- **COA-Tuned AE:** $TN = 12,777$, $FP = 736$, $FN = 427$, $TP = 1,164$

---

## 4. In-Depth Methodological Discussion

### 4.1 Why GWO Outperformed Other Metaheuristics
The Grey Wolf Optimizer's leadership hierarchy ($\alpha, \beta, \delta$) and social encircling mechanism effectively navigated the non-convex hyperparameter landscape. GWO discovered that:
1. **Bottleneck Width:** A wider latent bottleneck (`latent_dim = 32`) retained subtle morphological features that narrower bottlenecks compressed away.
2. **Denoising Regularization:** Injecting moderate Gaussian noise ($\sigma = 0.1157$) forced the model to reconstruct clean features, driving normal validation MSE down to `0.007546` (3.0x lower than the baseline's `0.022756`).
3. **Calibrated Threshold:** The lower normal reconstruction error produced a tighter decision boundary ($\theta = 0.016308$), enabling the detection of subtle arrhythmias that evaded the baseline gate.

### 4.2 Trade-Off Analysis: GWO vs. COA vs. Baseline
- **Diagnostic Performance:** GWO-AE is the optimal choice when maximizing abnormal beat detection is paramount, capturing 78.50% of arrhythmias.
- **Inference Speed & Resource Utilization:** COA-AE is the optimal choice for resource-constrained edge microcontrollers, requiring only 9,280 parameters and delivering 4.92 million inferences/second.
- **Simplicity & Zero-Tuning Cost:** Baseline AE remains a viable zero-cost reference that delivers competitive specificity (94.91%).

---

## 5. Explainability & Feature Contribution Analysis

By examining feature-wise squared reconstruction errors:
$$\Delta \text{MSE}_j = \mathbb{E}_{x \in \mathcal{D}_{abn}} [(x_j - \hat{x}_j)^2] - \mathbb{E}_{x \in \mathcal{D}_{norm}} [(x_j - \hat{x}_j)^2]$$
the primary features driving anomaly gating decisions were identified:
1. **Pre-RR Interval (Lead II):** Rhythmic prematurity in ventricular and supraventricular ectopy.
2. **Post-RR Interval (Lead II):** Compensatory pause following ectopic firing.
3. **QRS Duration (Lead II):** Broadening of the QRS complex characteristic of ventricular ectopic beats (VEB).
4. **QRS Morphological Features (`qrs_morph1`, `qrs_morph2` in Lead V):** Aberrant ventricular depolarization vectors.

---

## 6. Robustness & Streaming Evaluation

1. **Noise Robustness:** Under controlled additive Gaussian noise up to $\sigma = 0.25$, GWO-AE and PSO-AE preserved AUROC $> 0.88$, demonstrating the regularizing benefit of training with input corruption.
2. **Sensor Drift Simulation:** Simulated DC baseline drift on Lead II revealed that while AUROC remained stable, the False Positive Rate increased when fixed thresholds were applied without baseline tracking, highlighting the clinical value of continuous baseline recalibration in production telemetry.
3. **Streaming Scalability:** Inference throughput scaled from 70,400 events/second in single-sample streaming ($B=1$) to 1.90 million events/second under batch processing ($B=512$).

---

## 7. Artifact Directory Structure

All experimental data, checkpoints, and logs are persisted under structured directories:
```
artifacts/experiments/
├── baseline_ae/
│   ├── run_20261024_074848/
│   │   ├── best_model.pt
│   │   ├── config.json
│   │   ├── epoch_history.csv
│   │   ├── metrics.json
│   │   ├── runtime_metadata.json
│   │   ├── test_predictions.csv
│   │   └── val_predictions.csv
│   └── latest/ (mirrored)
├── pso_ae/
│   └── run_20261024_075128/ (includes search_history.json)
├── gwo_ae/
│   └── run_20261024_075517/ (includes search_history.json)
└── coa_ae/
    └── run_20261024_080158/ (includes search_history.json)

reports/
├── IMPLEMENTATION_AUDIT.md
├── FINAL_EXPERIMENT_REPORT.md
├── logs/
│   ├── baseline_ae.md
│   ├── pso_ae.md
│   ├── gwo_ae.md
│   ├── coa_ae.md
│   ├── experiment_comparison.md
│   └── visualization_index.md
├── tables/
│   ├── all_experiments_summary.json
│   └── model_comparison_table.csv
└── figures/ (16 publication PNGs and vector PDFs)
```

---

## 8. Dashboard Integration & Launch Verification

The interactive React dashboard has been updated:
- **Model Selector:** Toggles between Baseline AE, PSO-AE, GWO-AE, and COA-AE.
- **Side-by-Side Comparison:** Interactive metrics table displaying validation and test performance.
- **Embedded Research Figures:** Direct access to ROC curves, PR curves, convergence trajectories, and feature explainability plots.
- **Launch Command:**
  ```bash
  python3 demo/server.py
  ```
  Accessible in any standard browser at `http://localhost:8000`.
