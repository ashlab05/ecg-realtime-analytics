# Experiment Log: Particle Swarm Optimization Autoencoder (`pso_ae`)

## 1. Experiment Overview & Status
- **Experiment ID:** `pso_ae`
- **Run ID:** `run_20261024_075128`
- **Status:** `COMPLETED` (Success)
- **Artifacts Path:** `artifacts/experiments/pso_ae/run_20261024_075128/`
- **Optimization Strategy:** Particle Swarm Optimization (PSO) outer loop for hyperparameter search; Adam inner loop for weight backpropagation.

## 2. Execution Date, Time & Environment
- **Execution Timestamp:** `2026-10-24T07:51:28Z` (UTC) / `2026-10-24 13:21:28 IST`
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

## 5. Outer-Loop Metaheuristic Setup (PSO)
- **Swarm Size ($N$):** 6 particles
- **Iterations ($T$):** 5 iterations
- **Total Candidate Evaluations:** 36 evaluations (6 init + 30 iterative)
- **Inertia Weight ($w$):** Linearly decaying from $0.90$ to $0.40$
- **Acceleration Coefficients:** Cognitive $c_1 = 1.50$, Social $c_2 = 1.50$
- **Velocity Limit:** $v_{max} = 0.20$
- **Candidate Evaluation Budget:** 15 epochs per candidate with early stopping patience = 3
- **Fitness Criterion:** Validation AUPRC (Average Precision on binary anomaly detection)
- **Total Optimization Time:** `108.72 s`

### 5.1 Optimization Search History & Trajectory
| Iteration | Best Val AUPRC | Mean Val AUPRC | Best Particle Parameters Found |
| :---: | :---: | :---: | :--- |
| Init (0) | `0.68145` | `0.64684` | `latent=24, h=(128, 64), lr=0.00442` |
| 1 | `0.72497` | `0.68404` | `latent=16, h=(128, 64), lr=0.00125, noise=0.10` |
| 2 | `0.72856` | `0.70425` | `latent=24, h=(192, 96), lr=0.00350, noise=0.15` |
| **3** | **0.73165** | `0.70868` | `latent=24, h=(192, 96), lr=0.00416, noise=0.18` (Global Best) |
| 4 | `0.73165` | `0.71912` | Swarm converging towards gbest |
| 5 | `0.73165` | `0.72382` | Final swarm convergence |

## 6. Selected Best Model Configuration
- **Model Architecture:** Denoising Autoencoder (`noise_std = 0.1806`)
- **Latent Dimension:** `24`
- **Hidden Dimensions:** `(192, 96)`
- **Encoder:** `32 -> 192 -> 96 -> 24`
- **Decoder:** `24 -> 96 -> 192 -> 32`
- **Total Trainable Parameters:** 54,944 parameters
- **Learning Rate:** `0.004158` (Adam)
- **Weight Decay:** `2.53e-5`
- **Dropout Rate:** `0.1026`
- **Denoising Noise Std ($\sigma$):** `0.1806`
- **Batch Size:** `512`

## 7. Full Model Training Dynamics
- **Epochs Completed:** 27 (Early stopping triggered at epoch 27)
- **Best Epoch:** Epoch 17
- **Best Training Loss (MSE):** `0.057346`
- **Best Validation Normal Loss (MSE):** `0.024133`
- **Total Training Duration:** `7.73 s` (`~0.286 s/epoch`)

| Epoch | Train Loss (MSE) | Val Normal Loss (MSE) | Epoch Duration (s) | Cumulative Elapsed (s) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.230995 | 0.074931 | 0.278 | 0.278 |
| 5 | 0.069970 | 0.032142 | 0.275 | 1.102 |
| 10 | 0.063231 | 0.031187 | 0.274 | 2.472 |
| 15 | 0.062294 | 0.030426 | 0.291 | 3.926 |
| **17** | **0.057346** | **0.024133** | **0.275** | **4.476** |
| 20 | 0.059895 | 0.030751 | 0.275 | 5.300 |
| 25 | 0.062611 | 0.026097 | 0.302 | 6.810 |
| 27 | 0.058914 | 0.026514 | 0.274 | 7.732 |

## 8. Threshold Calibration
- **Calibration Method:** 95th percentile of Normal Validation reconstruction errors ($N=13,513$)
- **Calibrated Threshold:** `0.053568`
- **Score Statistics on Normal Val:**
  - Mean Error: `0.024133`
  - Std Error: `0.094105`
  - Median ($p_{50}$): `0.014610`
  - 95th Percentile ($p_{95}$): `0.053568`
  - 99th Percentile ($p_{99}$): `0.138701`
  - Max Error: `5.206325`

## 9. Performance Evaluation

### 9.1 Validation Split Metrics ($N=15,103$)
- **AUROC:** `0.949205`
- **AUPRC:** `0.738471`
- **Accuracy:** `0.931740` (93.17%)
- **Precision:** `0.646444` (64.64%)
- **Recall / Sensitivity:** `0.776870` (77.69%)
- **Specificity:** `0.949974` (95.00%)
- **F1-Score:** `0.705681` (70.57%)
- **Balanced Accuracy:** `0.863422` (86.34%)
- **Matthews Correlation Coefficient (MCC):** `0.671047`
- **False Positive Rate (FPR):** `0.050026` (5.00%)
- **FP per 1,000 Normal Samples:** `50.0`
- **Validation Confusion Matrix:**
  - True Negative ($TN$): `12,837`
  - False Positive ($FP$): `676`
  - False Negative ($FN$): `355`
  - True Positive ($TP$): `1,236`

### 9.2 Held-Out Test Split Metrics ($N=15,104$)
- **AUROC:** `0.945095`
- **AUPRC:** `0.728932`
- **Accuracy:** `0.927767` (92.78%)
- **Precision:** `0.629400` (62.94%)
- **Recall / Sensitivity:** `0.764299` (76.43%)
- **Specificity:** `0.947014` (94.70%)
- **F1-Score:** `0.690321` (69.03%)
- **Balanced Accuracy:** `0.855657` (85.57%)
- **Matthews Correlation Coefficient (MCC):** `0.653796`
- **False Positive Rate (FPR):** `0.052986` (5.30%)
- **FP per 1,000 Normal Samples:** `53.0`
- **Test Confusion Matrix:**
  - True Negative ($TN$): `12,797`
  - False Positive ($FP$): `716`
  - False Negative ($FN$): `375`
  - True Positive ($TP$): `1,216`

## 10. Computational Efficiency & Latency
- **Mean Latency per Sample:** `0.573 µs` (`0.000573 ms`)
- **Median Latency ($p_{50}$):** `0.538 µs`
- **95th Percentile Latency ($p_{95}$):** `0.649 µs`
- **99th Percentile Latency ($p_{99}$):** `0.655 µs`
- **Batch Throughput:** `1,920,051 events/second`

## 11. Visualizations
- `reports/figures/optimizer_convergence.png`: Orange curve illustrating PSO swarm fitness progressing from 0.68145 to 0.73165.
- `reports/figures/roc_curves_comparison.png`: Orange trace with 0.9451 AUROC.
- `reports/figures/pr_curves_comparison.png`: Orange trace with 0.7289 AUPRC.
- `reports/figures/confusion_matrices_all.png`: Subplot 2 showing test confusion matrix.

## 12. Reproduction Command
```bash
python3 training/run_experiments.py --only pso_ae --n-agents 6 --n-iterations 5 --seed 42 --threshold-percentile 95.0
```

## 13. Scientific Interpretation
Particle Swarm Optimization successfully tuned the Autoencoder, uncovering that a wider hidden structure (`192 -> 96 -> 24`) coupled with a moderate input corruption level (`noise_std = 0.1806`, Denoising AE) enhances test AUROC from 0.9404 to 0.9451. Early stopping at epoch 17 prevented overfitting, reducing training duration to 7.73 seconds.
