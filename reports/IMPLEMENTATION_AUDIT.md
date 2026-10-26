# ECG Anomaly Detection: Implementation Audit & Technical Review

**Document Version:** 1.0  
**Date:** 2026-10-24  
**Project:** ECG Real-Time Arrhythmia Analytics  
**Repository:** `realtime/ecg-realtime-analytics`  

---

## 1. Executive Summary

This audit inspects the current state of the ECG anomaly detection repository. The system was originally conceived as a two-stage hybrid architecture:
1. **Stage 1 (Deep Learning Autoencoder Gate):** High-throughput anomaly filter trained exclusively on Normal ($N$) heartbeats. Beats with reconstruction error exceeding a calibrated threshold are flagged as anomalous.
2. **Stage 2 (Supervised Arrhythmia Classifier):** Multi-class XGBoost classifier assigning specific arrhythmia labels (`SVEB`, `VEB`, `F`, `Q`) to flagged anomalous heartbeats.

The primary objective of the current research phase is to **rigorously evaluate and compare metaheuristic hyperparameter optimization techniques for the Stage 1 Autoencoder**:
- **Baseline AE + Adam** (default architecture and learning rate).
- **PSO-tuned AE + Adam** (Particle Swarm Optimization outer loop).
- **GWO-tuned AE + Adam** (Grey Wolf Optimizer outer loop).
- **COA-tuned AE + Adam** (Coati Optimization Algorithm outer loop).

All metaheuristics operate strictly as **outer-loop hyperparameter optimizers** selecting architectural and training parameters, while **Adam** remains the gradient-based weight optimizer.

---

## 2. Repository Structure & Artifact Inventory

### 2.1 Directory Structure
```
ecg-realtime-analytics/
├── api/                  # FastAPI REST backend & Kafka consumers
├── dashboard/            # React 18 + Vite + TailwindCSS + Recharts dashboard
├── data/
│   ├── raw/              # MIT-BIH Arrhythmia Database.csv (100,689 rows, 34 columns)
│   └── processed/        # Placeholder for processed splits
├── demo/
│   └── server.py         # Self-contained demo server (FastAPI + WebSocket + static UI)
├── docs/
│   ├── figures/          # EDA plots (class distribution, RR-interval, correlation matrix)
│   ├── REDIS_STREAMS.md  # Alternative broker documentation
│   └── TECHNICAL.md      # Architectural design specifications
├── ecg_core/             # Shared core package
│   ├── config.py         # Central configuration and paths
│   ├── features.py       # 32 canonical feature definitions (16 Lead II, 16 Lead V)
│   ├── labels.py         # AAMI-compliant classes (N, SVEB, VEB, F, Q) & binary mapping
│   ├── metrics.py        # Classification, latency, and experiment persistence utilities
│   ├── models/           # PyTorch ECGAutoencoder and TabularMLP implementations
│   ├── preprocessing.py  # Validation, scaling, and train/val/test split routines
│   └── training_utils.py # EarlyStopping, EpochLogger, seed_everything
├── logs/
│   └── training/         # Historical training logs and experiment JSONs
├── models/
│   ├── autoencoder/      # Trained autoencoder.pt and autoencoder_meta.json
│   ├── preprocessing/    # Fitted StandardScaler (scaler.joblib) and feature_metadata.json
│   └── xgboost/          # Saved Stage-2 XGBoost model and metadata
├── report/               # IEEE proposal, report generator, and 14 system figures
├── tests/                # 42 unit and integration tests (all passing)
└── training/             # Standalone training and evaluation scripts
```

### 2.2 Existing Model Artifacts & Historical Results
| Artifact | Path | Size | Status & Content |
| :--- | :--- | :--- | :--- |
| **Autoencoder Weights** | `models/autoencoder/autoencoder.pt` | 113 KB | PyTorch state dict (32 → 128 → 64 → 16 → 64 → 128 → 32) |
| **Autoencoder Meta** | `models/autoencoder/autoencoder_meta.json` | 397 B | `latent_dim`: 16, `threshold`: 0.059017 ($p_{95}$ of normal val error) |
| **Standard Scaler** | `models/preprocessing/scaler.joblib` | 1.38 KB | Scikit-learn `StandardScaler` fit on training split |
| **Feature Metadata** | `models/preprocessing/feature_metadata.json` | 799 B | 32 canonical features, class names, and index maps |
| **Historical AE Log** | `logs/training/autoencoder_training_log.csv` | 2.17 KB | 50 epochs; initial loss: 0.3338, final val loss: 0.0230 |
| **Ablation Summary** | `logs/training/experiments/comparison_summary.json` | 2.25 KB | Ablation results for Experiments A–E |

---

## 3. Dataset & Preprocessing Inspection

### 3.1 Feature Dimensions & Verification
- **Raw File:** `MIT-BIH Arrhythmia Database.csv` (45.48 MB).
- **Total Rows:** 100,689 samples.
- **Total Columns:** 34 columns consisting of:
  - `record`: Patient/ECG recording identifier (44 unique records).
  - `type`: Ground truth arrhythmia class.
  - **32 Morphological and Interval Features:**
    - 16 features from Lead II (`0_pre-RR`, `0_post-RR`, `0_pPeak`, `0_tPeak`, `0_rPeak`, `0_sPeak`, `0_qPeak`, `0_qrs_interval`, `0_pq_interval`, `0_qt_interval`, `0_st_interval`, `0_qrs_morph0` to `0_qrs_morph4`).
    - 16 features from Lead V (`1_pre-RR`, `1_post-RR`, `1_pPeak`, `1_tPeak`, `1_rPeak`, `1_sPeak`, `1_qPeak`, `1_qrs_interval`, `1_pq_interval`, `1_qt_interval`, `1_st_interval`, `1_qrs_morph0` to `1_qrs_morph4`).
- **Critical Finding:** The data consists of **pre-extracted feature vectors per beat**, **not raw time-series ECG waveforms**. All reconstruction analyses and visualizations must explicitly reflect feature reconstructions.

### 3.2 Class Distribution & Binary Anomaly Mapping
The dataset adheres to the AAMI EC57 standard grouped into 5 classes:
- **Normal ($N$):** 90,083 beats (89.47%) $\rightarrow$ `binary_label = 0` (Negative)
- **Ventricular Ectopic Beat ($VEB$):** 7,009 beats (6.96%) $\rightarrow$ `binary_label = 1` (Positive / Anomaly)
- **Supraventricular Ectopic Beat ($SVEB$):** 2,779 beats (2.76%) $\rightarrow$ `binary_label = 1` (Positive / Anomaly)
- **Fusion Beat ($F$):** 803 beats (0.80%) $\rightarrow$ `binary_label = 1` (Positive / Anomaly)
- **Unknown Beat ($Q$):** 15 beats (0.015%) $\rightarrow$ `binary_label = 1` (Positive / Anomaly)
- **Total Anomalies:** 10,606 beats (10.53%).
- **Class Imbalance Ratio:** 8.5 : 1 (Normal to Abnormal).

### 3.3 Data Leakage Analysis & Splitting Strategy
1. **Stratified Splitting (`strategy="stratified"`):**
   - 70% Train (70,482 beats), 15% Validation (15,103 beats), 15% Test (15,104 beats).
   - **Risk of Data Leakage:** Because beats from the same patient/record appear in both train and test partitions, models can memorize patient-specific baseline morphologies rather than generalizable arrhythmic pathology.
2. **Record-Aware Splitting (`strategy="record_aware"`):**
   - Implemented in `ecg_core/preprocessing.py`, holding out entire records so no patient appears across splits.
   - **Empirical Challenge with Rare Classes:** The 15 $Q$ beats appear in only 5 records (101, 105, 203, 208, 214). The 803 $F$ beats are heavily concentrated in records 208 (373 beats) and 213 (362 beats). A naive random record holdout can leave the validation or test split with 0 instances of $Q$ or $F$.
3. **Audit Verification:**
   - Preprocessing scaling (`StandardScaler`) is **strictly fitted on the training split only** (`X[idx_train]`), ensuring no leakage of test/validation statistics into model normalization.
   - To maintain consistency with the existing baseline and benchmark literature while documenting the limitation, the primary benchmark protocol uses the fixed stratified split (`seed=42`), while patient-level generalization is audited as a dedicated robustness dimension.

---

## 4. Current Implementation vs. Missing Requirements

| Category | Component | Current State | Required Work |
| :--- | :--- | :--- | :--- |
| **Model** | Standard Autoencoder | Implemented (32 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 16) | Preserve as Baseline AE |
| **Model** | Denoising Autoencoder | Not explicitly parameterized with noise injection | Add configurable Gaussian input noise $\sigma$ |
| **Optimization** | Adam Backprop | Implemented for default model | Keep as inner-loop weight optimizer |
| **Optimization** | Metaheuristics (PSO, GWO, COA) | **Completely missing** | Implement outer-loop PSO, GWO, and COA with unified budget |
| **Model Selection** | Fitness Objective | Val F1 used informally in training loop | Formalize **Validation AUPRC** as the primary selection criterion |
| **Threshold** | Calibration Policy | $p_{95}$ of normal val error | Formalize and evaluate $p_{95}$ and $p_{99}$ policies |
| **Metrics** | Comprehensive Reporting | Precision/Recall/F1/Latency present | Add AUROC, AUPRC, Specificity, MCC, FPR, FP per 1,000 |
| **Visualizations** | Publication-Quality Plots | Only 3 EDA plots and generic report figures | Implement full suite in `reports/figures/` (ROC, PR, distributions, explanations) |
| **Dashboard** | Model Comparison UI | Shows ablations A–E, lacks metaheuristic selector | Add multi-model selector (Baseline, PSO, GWO, COA), side-by-side tables, curves |
| **Documentation** | Detailed Markdown Logs | Fragmented JSONs in `logs/` | Generate standardized logs under `reports/logs/` and `reports/FINAL_EXPERIMENT_REPORT.md` |

---

## 5. Proposed Execution Plan

1. **Step 1 — Model & Denoising Architecture Extension:**
   - Update `ECGAutoencoder` to support flexible hidden layers, latent dimensions, dropout, and training-time noise injection (`noise_std`).
2. **Step 2 — Metaheuristic Optimization Framework:**
   - Implement `training/metaheuristics/`:
     - Unified parameter search space (latent dimension, hidden width, learning rate, weight decay, dropout, noise level, batch size).
     - Standardized candidate evaluation harness using Validation AUPRC.
     - Particle Swarm Optimization (PSO).
     - Grey Wolf Optimizer (GWO).
     - Coati Optimization Algorithm (COA).
     - Enforce identical evaluation budgets ($N=6$ population, $T=5$ iterations = 30 evaluations per optimizer).
3. **Step 3 — Execute Experiments & Persist Artifacts:**
   - Run Baseline AE, PSO-AE, GWO-AE, and COA-AE.
   - Save weights, configs, epoch logs, sample predictions, and metrics under `artifacts/experiments/<model>/<run_id>/`.
4. **Step 4 — Metrics & Statistical Analysis:**
   - Calculate all 15 required metrics for validation and test splits with explicit zero-division policy (0.0).
5. **Step 5 — Publication Visualizations:**
   - Generate high-resolution figures in `reports/figures/` (Loss curves, convergence trajectories, ROC/PR curves, confusion matrices, error distributions, and SHAP/feature-error explanations).
6. **Step 6 — Dashboard Enhancements:**
   - Update `demo/server.py` to expose the new experiment artifacts.
   - Update `ModelPerformancePage.tsx` with a multi-model selector, side-by-side metric tables, and interactive diagnostic charts.
   - Rebuild React frontend (`npm run build`).
7. **Step 7 — Documentation & Verification:**
   - Author all required Markdown logs in `reports/logs/` and `reports/FINAL_EXPERIMENT_REPORT.md`.
   - Run end-to-end verification and deliver final summary.
