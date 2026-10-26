# Experiment Log: Grey Wolf Optimizer Autoencoder (`gwo_ae`)

## 1. Experiment Overview & Status
- **Experiment ID:** `gwo_ae`
- **Run ID:** `run_20261024_075517`
- **Status:** `COMPLETED` (Success — **Top Overall Performing Model**)
- **Artifacts Path:** `artifacts/experiments/gwo_ae/run_20261024_075517/`
- **Optimization Strategy:** Grey Wolf Optimizer (GWO) outer loop for hyperparameter search; Adam inner loop for weight backpropagation.

## 2. Execution Date, Time & Environment
- **Execution Timestamp:** `2026-10-24T07:55:17Z` (UTC) / `2026-10-24 13:25:17 IST`
- **Host Platform:** macOS (Darwin 24.3.0, ARM64 / Apple Silicon)
- **Python Version:** `3.13.9`
- **PyTorch Version:** `2.10.0` (Device: `cpu`)
- **Scikit-Learn Version:** `1.6.1`
- **Deterministic Random Seed:** `42`
- **Git Commit:** `854ba24` (`master`)

## 3. Dataset Characteristics & Binary Label Mapping
- **Dataset Path:** `data/raw/MIT-BIH Arrhythmia Database.csv`
- **Total Dataset Size:** 100,689 samples, 34 columns (record, type, 32 numerical features)
- **Feature Space:** 32 engineered features (16 Lead II, 16 Lead V).
- **Binary Mapping:** Normal (`binary_label = 0`, 90,083 beats), Abnormal (`binary_label = 1`, 10,606 beats).

## 4. Preprocessing & Partitioning
- **Split Strategy:** Stratified split (70/15/15, `seed=42`).
  - **Train:** 70,482 beats (Normal: 63,057, Abnormal: 7,425)
  - **Validation:** 15,103 beats (Normal: 13,513, Abnormal: 1,590)
  - **Test:** 15,104 beats (Normal: 13,513, Abnormal: 1,591)
- **StandardScaler:** Fit exclusively on training split.

## 5. Outer-Loop Metaheuristic Setup (GWO)
- **Pack Size ($N$):** 6 wolves ($\alpha, \beta, \delta, \omega_1, \omega_2, \omega_3$)
- **Iterations ($T$):** 5 iterations
- **Total Candidate Evaluations:** 36 evaluations (6 init + 30 iterative)
- **Hunting Parameter ($a$):** Linearly decreasing from $2.0$ to $0.0$ ($a = 2 \cdot (1 - t/T)$)
- **Encircling Vector:** $A = 2 a r_1 - a$, $C = 2 r_2$, where $r_1, r_2 \sim U(0, 1)$
- **Position Update Rule:** Toward $(\vec{X}_1 + \vec{X}_2 + \vec{X}_3) / 3$
- **Candidate Evaluation Budget:** 15 epochs per candidate with early stopping patience = 3
- **Fitness Criterion:** Validation AUPRC (Average Precision on binary anomaly detection)
- **Total Optimization Time:** `139.63 s`

### 5.1 Optimization Search History & Trajectory
| Iteration | Alpha Fitness (Best Val AUPRC) | Mean Pack Fitness | Alpha Configuration Found |
| :---: | :---: | :---: | :--- |
| Init (0) | `0.68145` | `0.64684` | `latent=24, h=(128, 64), lr=0.00442` |
| 1 | `0.72454` | `0.69796` | `latent=16, h=(128, 64), lr=0.00125` |
| 2 | `0.73328` | `0.71321` | `latent=32, h=(192, 96), lr=0.00180, noise=0.10` |
| **3** | **0.74527** | `0.72535` | `latent=32, h=(192, 96), lr=0.00126, noise=0.12` (**Global Best**) |
| 4 | `0.74527` | `0.73708` | Wolves encircling the alpha optimum |
| 5 | `0.74527` | `0.73114` | Pack fully converged |

## 6. Selected Best Model Configuration
- **Model Architecture:** Denoising Autoencoder (`noise_std = 0.1157`)
- **Latent Dimension:** `32` (Equal to input dimension, enabling rich bottleneck representation)
- **Hidden Dimensions:** `(192, 96)`
- **Encoder:** `32 -> 192 -> 96 -> 32`
- **Decoder:** `32 -> 96 -> 192 -> 32`
- **Total Trainable Parameters:** 57,280 parameters
- **Learning Rate:** `0.001258` (Adam)
- **Weight Decay:** `2.20e-6`
- **Dropout Rate:** `0.0337`
- **Denoising Noise Std ($\sigma$):** `0.1157`
- **Batch Size:** `512`

## 7. Full Model Training Dynamics
- **Epochs Completed:** 50
- **Best Epoch:** Epoch 50
- **Best Training Loss (MSE):** `0.022346`
- **Best Validation Normal Loss (MSE):** `0.007546`
- **Total Training Duration:** `14.25 s` (`~0.285 s/epoch`)

| Epoch | Train Loss (MSE) | Val Normal Loss (MSE) | Epoch Duration (s) | Cumulative Elapsed (s) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.288570 | 0.085610 | 0.289 | 0.289 |
| 5 | 0.039736 | 0.017541 | 0.280 | 1.127 |
| 10 | 0.030659 | 0.011507 | 0.282 | 2.520 |
| 15 | 0.027196 | 0.009819 | 0.284 | 3.937 |
| 20 | 0.025734 | 0.009559 | 0.281 | 5.330 |
| 25 | 0.025066 | 0.009309 | 0.280 | 6.738 |
| 30 | 0.024052 | 0.008281 | 0.285 | 8.263 |
| 35 | 0.023373 | 0.008366 | 0.283 | 9.661 |
| 40 | 0.023164 | 0.008177 | 0.284 | 11.067 |
| 45 | 0.022486 | 0.008776 | 0.283 | 12.520 |
| **50** | **0.022346** | **0.007546** | **0.284** | **14.250** |

## 8. Threshold Calibration
- **Calibration Method:** 95th percentile of Normal Validation reconstruction errors ($N=13,513$)
- **Calibrated Threshold:** `0.016308`
- **Score Statistics on Normal Val:**
  - Mean Error: `0.007546`
  - Std Error: `0.035890`
  - Median ($p_{50}$): `0.004789`
  - 95th Percentile ($p_{95}$): `0.016308`
  - 99th Percentile ($p_{99}$): `0.041557`
  - Max Error: `3.533689`

## 9. Performance Evaluation

### 9.1 Validation Split Metrics ($N=15,103$)
- **AUROC:** `0.948330`
- **AUPRC:** `0.772577` (Top validation AUPRC among all models)
- **Accuracy:** `0.933197` (93.32%)
- **Precision:** `0.650465` (65.05%)
- **Recall / Sensitivity:** `0.790698` (79.07%)
- **Specificity:** `0.949974` (95.00%)
- **F1-Score:** `0.713759` (71.38%)
- **Balanced Accuracy:** `0.870336` (87.03%)
- **Matthews Correlation Coefficient (MCC):** `0.680480`
- **False Positive Rate (FPR):** `0.050026` (5.00%)
- **FP per 1,000 Normal Samples:** `50.0`
- **Validation Confusion Matrix:**
  - True Negative ($TN$): `12,837`
  - False Positive ($FP$): `676`
  - False Negative ($FN$): `333`
  - True Positive ($TP$): `1,258`

### 9.2 Held-Out Test Split Metrics ($N=15,104$)
- **AUROC:** `0.946602` (**Top Test AUROC** across all experiments)
- **AUPRC:** `0.761236` (**Top Test AUPRC**, +2.67% improvement over Baseline)
- **Accuracy:** `0.928297` (92.83%)
- **Precision:** `0.627638` (62.76%)
- **Recall / Sensitivity:** `0.785041` (**Top Abnormal Recall: 78.50%**, +1.82% over Baseline)
- **Specificity:** `0.945164` (94.52%)
- **F1-Score:** `0.697571` (69.76%)
- **Balanced Accuracy:** `0.865102` (86.51%)
- **Matthews Correlation Coefficient (MCC):** `0.662768` (**Top Test MCC**)
- **False Positive Rate (FPR):** `0.054836` (5.48%)
- **FP per 1,000 Normal Samples:** `54.8`
- **Test Confusion Matrix:**
  - True Negative ($TN$): `12,772`
  - False Positive ($FP$): `741`
  - False Negative ($FN$): `342`
  - True Positive ($TP$): `1,249`

## 10. Computational Efficiency & Latency
- **Mean Latency per Sample:** `0.551 µs` (`0.000551 ms`)
- **Median Latency ($p_{50}$):** `0.556 µs`
- **95th Percentile Latency ($p_{95}$):** `0.563 µs`
- **99th Percentile Latency ($p_{99}$):** `0.563 µs`
- **Batch Throughput:** `1,895,434 events/second`

## 11. Visualizations
- `reports/figures/optimizer_convergence.png`: Green curve demonstrating GWO finding Alpha fitness of 0.74527 at iteration 3.
- `reports/figures/roc_curves_comparison.png`: Green trace demonstrating highest AUROC of 0.9466.
- `reports/figures/pr_curves_comparison.png`: Green trace demonstrating superior precision-recall envelope (AUPRC = 0.7612).
- `reports/figures/confusion_matrices_all.png`: Subplot 3 with 1,249 detected abnormal beats.
- `reports/figures/reconstruction_error_distribution.png`: Clean separation between Normal and Abnormal distributions with threshold 0.0163.
- `reports/figures/top_contributing_features.png`: Diagnostic features identified via GWO error discrepancy.

## 12. Reproduction Command
```bash
python3 training/run_experiments.py --only gwo_ae --n-agents 6 --n-iterations 5 --seed 42 --threshold-percentile 95.0
```

## 13. Scientific Interpretation
The Grey Wolf Optimizer achieved the highest discrimination power across all evaluated models. By selecting an expanded bottleneck (`latent_dim = 32`) combined with a moderate denoising level (`noise_std = 0.1157`) and gentle regularization (`dropout = 0.0337`, `weight_decay = 2.20e-6`), GWO effectively learned sharp normal feature representations. Consequently, Normal validation MSE decreased to `0.0075`, allowing a tight calibrated threshold (`0.0163`) that captured 78.50% of test arrhythmic heartbeats while sustaining ~1.9 million events/sec throughput.
