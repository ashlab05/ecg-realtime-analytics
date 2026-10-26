# Experiment Log: Baseline Autoencoder + Adam (`baseline_ae`)

## 1. Experiment Overview & Status
- **Experiment ID:** `baseline_ae`
- **Run ID:** `run_20261024_074848`
- **Status:** `COMPLETED` (Success)
- **Artifacts Path:** `artifacts/experiments/baseline_ae/run_20261024_074848/`
- **Execution Role:** Stage 1 Anomaly Gate Baseline (un-tuned default reference)

## 2. Execution Date, Time & Environment
- **Execution Timestamp:** `2026-10-24T07:48:48Z` (UTC) / `2026-10-24 13:18:48 IST`
- **Host Platform:** macOS (Darwin 24.3.0, ARM64 / Apple Silicon)
- **Python Version:** `3.13.9`
- **PyTorch Version:** `2.10.0` (Device: `cpu`)
- **Scikit-Learn Version:** `1.6.1`
- **Deterministic Random Seed:** `42` (`seed_everything(42)`)
- **Git Commit:** `854ba24` (`master`)

## 3. Dataset Characteristics & Binary Label Mapping
- **Dataset Path:** `data/raw/MIT-BIH Arrhythmia Database.csv`
- **Total Dataset Size:** 100,689 samples, 34 columns (record, type, 32 numerical features)
- **Feature Space:** 32 engineered morphological and interval features (16 Lead II, 16 Lead V). Data represents pre-extracted beat features, not raw continuous waveforms.
- **Original Multi-Class Distribution:**
  - `N` (Normal): 90,083 (89.467%)
  - `VEB` (Ventricular Ectopic Beat): 7,009 (6.961%)
  - `SVEB` (Supraventricular Ectopic Beat): 2,779 (2.760%)
  - `F` (Fusion Beat): 803 (0.798%)
  - `Q` (Unknown Beat): 15 (0.015%)
- **Binary Anomaly Mapping:**
  - Negative Class (Normal, `binary_label = 0`): `type == "N"` (90,083 samples)
  - Positive Class (Abnormal / Anomaly, `binary_label = 1`): `type in ["VEB", "SVEB", "F", "Q"]` (10,606 samples)
  - Class Imbalance: 8.5 : 1 (Normal : Abnormal)

## 4. Preprocessing & Partitioning
- **Split Strategy:** Stratified split (70% train, 15% validation, 15% test; `seed=42`).
  - **Train Set:** 70,482 samples (Normal: 63,057, Abnormal: 7,425)
  - **Validation Set:** 15,103 samples (Normal: 13,513, Abnormal: 1,590)
  - **Test Set:** 15,104 samples (Normal: 13,513, Abnormal: 1,591)
- **Feature Scaling:** `StandardScaler` fitted **strictly on the training split only** (`X[idx_train]`). Scaler parameters (mean and standard deviation for 32 features) persisted to `models/preprocessing/scaler.joblib`.
- **Data Leakage Audit:** While scaling statistics do not leak across splits, beat-level stratified splitting permits beats from the same patient record to appear across partitions. This matches standard benchmarking protocol while noting intra-patient correlation as a documented limitation.

## 5. Model Architecture & Parameters
- **Architecture Family:** Fully-connected standard Autoencoder (`ECGAutoencoder`)
- **Input Dimension:** 32 features
- **Encoder:** `Linear(32, 128) -> ReLU -> Dropout(0.10) -> Linear(128, 64) -> ReLU -> Dropout(0.10) -> Linear(64, 16)`
- **Bottleneck (Latent Space):** 16 dimensions
- **Decoder:** `Linear(16, 64) -> ReLU -> Dropout(0.10) -> Linear(64, 128) -> ReLU -> Dropout(0.10) -> Linear(128, 32)`
- **Total Trainable Parameters:** 26,080 parameters
- **Weight Optimizer:** Adam (`lr = 0.001`, `weight_decay = 0.0`, `betas = (0.9, 0.999)`)
- **Batch Size:** 256
- **Training Restriction:** Trained exclusively on Normal training samples ($N=63,057$).

## 6. Training Dynamics & Epoch History
- **Max Epochs:** 50
- **Early Stopping:** Patience = 10 (did not trigger; completed full 50 epochs)
- **Best Epoch:** Epoch 47
- **Best Training Loss (MSE):** `0.064469`
- **Best Validation Normal Loss (MSE):** `0.022756`
- **Total Training Duration:** `11.81 s` (`~0.236 s/epoch`)

| Epoch | Train Loss (MSE) | Val Normal Loss (MSE) | Epoch Duration (s) | Cumulative Elapsed (s) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.333754 | 0.128649 | 0.286 | 0.286 |
| 5 | 0.099937 | 0.047617 | 0.232 | 1.210 |
| 10 | 0.083082 | 0.036738 | 0.234 | 2.342 |
| 15 | 0.075850 | 0.031636 | 0.235 | 3.489 |
| 20 | 0.072582 | 0.030187 | 0.233 | 4.619 |
| 25 | 0.070211 | 0.028059 | 0.236 | 5.792 |
| 30 | 0.068486 | 0.027123 | 0.234 | 6.948 |
| 35 | 0.067160 | 0.025504 | 0.232 | 8.068 |
| 40 | 0.065486 | 0.025708 | 0.234 | 9.219 |
| 45 | 0.064749 | 0.023736 | 0.235 | 10.352 |
| **47** | **0.064469** | **0.022756** | **0.234** | **10.810** |
| 50 | 0.064751 | 0.022961 | 0.235 | 11.502 |

## 7. Hyperparameter Search History
- **Search Method:** None (Default Baseline Parameters)
- **Optimization Elapsed Time:** `0.0 s`
- **Candidate Evaluations:** `0`

## 8. Threshold Calibration Policy
- **Calibration Split:** Validation Normal samples ($N=13,513$)
- **Score Definition:** Mean squared reconstruction error across 32 features ($S = \frac{1}{32} \sum_{i=1}^{32} (x_i - \hat{x}_i)^2$)
- **Normal Validation Score Statistics:**
  - Mean Error: `0.022961`
  - Std Error: `0.083109`
  - Median ($p_{50}$): `0.013181`
  - 95th Percentile ($p_{95}$): `0.057829` (Calibrated Anomaly Decision Threshold)
  - 99th Percentile ($p_{99}$): `0.155999`
  - Maximum Error: `4.495484`

## 9. Validation & Test Performance Evaluation

### 9.1 Validation Split Metrics ($N=15,103$)
- **AUROC:** `0.942966`
- **AUPRC (Average Precision):** `0.750513`
- **Accuracy:** `0.930478` (93.05%)
- **Precision:** `0.642514` (64.25%)
- **Recall / Sensitivity:** `0.774214` (77.42%)
- **Specificity:** `0.948864` (94.89%)
- **F1-Score:** `0.702251` (70.23%)
- **Balanced Accuracy:** `0.861539` (86.15%)
- **Matthews Correlation Coefficient (MCC):** `0.667231`
- **False Positive Rate (FPR):** `0.051136` (5.11%)
- **FP per 1,000 Normal Samples:** `51.1`
- **Validation Confusion Matrix:**
  - True Negative ($TN$): `12,822`
  - False Positive ($FP$): `691`
  - False Negative ($FN$): `359`
  - True Positive ($TP$): `1,231`

### 9.2 Held-Out Test Split Metrics ($N=15,104$)
- **AUROC:** `0.940449`
- **AUPRC (Average Precision):** `0.734572`
- **Accuracy:** `0.929886` (92.99%)
- **Precision:** `0.639413` (63.94%)
- **Recall / Sensitivity:** `0.766813` (76.68%)
- **Specificity:** `0.949086` (94.91%)
- **F1-Score:** `0.697342` (69.73%)
- **Balanced Accuracy:** `0.857950` (85.80%)
- **Matthews Correlation Coefficient (MCC):** `0.661535`
- **False Positive Rate (FPR):** `0.050914` (5.09%)
- **FP per 1,000 Normal Samples:** `50.9`
- **Zero-Division Policy:** Handled explicitly with return value `0.0`
- **Test Confusion Matrix:**
  - True Negative ($TN$): `12,825`
  - False Positive ($FP$): `688`
  - False Negative ($FN$): `371`
  - True Positive ($TP$): `1,220`

## 10. Computational Efficiency & Latency
- **Inference Timing Methodology:** 5 repeat evaluations over held-out test split ($N=15,104$) using `time.perf_counter()`.
- **Mean Latency per Sample:** `0.357 µs` (`0.000357 ms`)
- **Median Latency ($p_{50}$):** `0.337 µs`
- **95th Percentile Latency ($p_{95}$):** `0.450 µs`
- **99th Percentile Latency ($p_{99}$):** `0.453 µs`
- **Batch Throughput:** `2,969,891 events/second`

## 11. Relevant Visualizations
- `reports/figures/loss_curves_comparison.png`: Blue trace showing baseline convergence across 50 epochs.
- `reports/figures/roc_curves_comparison.png`: Baseline ROC curve achieving 0.9404 AUROC.
- `reports/figures/pr_curves_comparison.png`: Baseline PR curve achieving 0.7346 AUPRC.
- `reports/figures/confusion_matrices_all.png`: Subplot 1 displaying 12,825 TN, 688 FP, 371 FN, and 1,220 TP.
- `reports/figures/metrics_bar_comparison.png`: Side-by-side bar chart comparison.

## 12. Reproduction Command
```bash
python3 training/run_experiments.py --only baseline_ae --seed 42 --threshold-percentile 95.0
```

## 13. Scientific Interpretation
The baseline Autoencoder demonstrates strong anomaly filtering capability, achieving 94.04% AUROC and correctly detecting 76.68% of abnormal beats while passing 94.91% of normal beats. Its throughput exceeding 2.9 million events/second validates the feasibility of using an autoencoder gate in real-time streaming architectures.
