# Experiment Log: Coati Optimization Algorithm Autoencoder (`coa_ae`)

## 1. Experiment Overview & Status
- **Experiment ID:** `coa_ae`
- **Run ID:** `run_20261024_080158`
- **Status:** `COMPLETED` (Success — **Top Inference Speed & Efficiency Champion**)
- **Artifacts Path:** `artifacts/experiments/coa_ae/run_20261024_080158/`
- **Optimization Strategy:** Coati Optimization Algorithm (COA) outer loop for hyperparameter search; Adam inner loop for weight backpropagation.

## 2. Execution Date, Time & Environment
- **Execution Timestamp:** `2026-10-24T08:01:58Z` (UTC) / `2026-10-24 13:31:58 IST`
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

## 5. Outer-Loop Metaheuristic Setup (COA)
- **Population Size ($N$):** 6 coatis
- **Iterations ($T$):** 5 iterations
- **Total Candidate Evaluations:** 36 evaluations (6 init + 30 iterative; budget strictly identical to PSO and GWO)
- **Phase 1 (Hunting / Exploration):** Tree-climbing coatis ($i \le N/2$) hunt toward the best agent ($\vec{X}_{best}$); ground coatis explore local ground regions.
- **Phase 2 (Evasion / Exploitation):** Local random walks within shrinking radius $R = 0.2 \cdot (1 - t/T)$.
- **Candidate Evaluation Budget:** 15 epochs per candidate with early stopping patience = 3
- **Fitness Criterion:** Validation AUPRC (Average Precision on binary anomaly detection)
- **Total Optimization Time:** `118.01 s`

### 5.1 Optimization Search History & Trajectory
| Iteration | Best Coati Fitness (Val AUPRC) | Mean Population Fitness | Best Coati Parameters Found |
| :---: | :---: | :---: | :--- |
| Init (0) | `0.68145` | `0.64684` | `latent=24, h=(128, 64), lr=0.00442` |
| 1 | `0.71657` | `0.65183` | `latent=16, h=(128, 64), lr=0.00125` |
| 2 | `0.71657` | `0.64936` | `latent=16, h=(128, 64), lr=0.00125` |
| 3 | `0.71657` | `0.65017` | `latent=16, h=(128, 64), lr=0.00125` |
| 4 | `0.71657` | `0.66808` | Exploitation around best compact basin |
| **5** | **0.72598** | `0.67253` | `latent=12, h=(64, 32), lr=0.00138, noise=0.11` (**Global Best**) |

## 6. Selected Best Model Configuration
- **Model Architecture:** Compact Denoising Autoencoder (`noise_std = 0.1120`)
- **Latent Dimension:** `12`
- **Hidden Dimensions:** `(64, 32)`
- **Encoder:** `32 -> 64 -> 32 -> 12`
- **Decoder:** `12 -> 32 -> 64 -> 32`
- **Total Trainable Parameters:** 9,280 parameters (lowest parameter footprint among all candidates)
- **Learning Rate:** `0.001383` (Adam)
- **Weight Decay:** `4.67e-5`
- **Dropout Rate:** `0.0329`
- **Denoising Noise Std ($\sigma$):** `0.1120`
- **Batch Size:** `128`

## 7. Full Model Training Dynamics
- **Epochs Completed:** 50
- **Best Epoch:** Epoch 50
- **Best Training Loss (MSE):** `0.061211`
- **Best Validation Normal Loss (MSE):** `0.026189`
- **Total Training Duration:** `12.38 s` (`~0.248 s/epoch`)

| Epoch | Train Loss (MSE) | Val Normal Loss (MSE) | Epoch Duration (s) | Cumulative Elapsed (s) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.277096 | 0.122480 | 0.245 | 0.245 |
| 5 | 0.096979 | 0.056276 | 0.248 | 1.237 |
| 10 | 0.077907 | 0.039410 | 0.247 | 2.472 |
| 15 | 0.072917 | 0.036108 | 0.246 | 3.702 |
| 20 | 0.069703 | 0.032975 | 0.245 | 4.932 |
| 25 | 0.067011 | 0.031851 | 0.247 | 6.166 |
| 30 | 0.065742 | 0.031142 | 0.249 | 7.411 |
| 35 | 0.065269 | 0.029815 | 0.246 | 8.641 |
| 40 | 0.062606 | 0.029195 | 0.246 | 9.871 |
| 45 | 0.061711 | 0.028468 | 0.247 | 11.106 |
| **50** | **0.061211** | **0.026189** | **0.248** | **12.380** |

## 8. Threshold Calibration
- **Calibration Method:** 95th percentile of Normal Validation reconstruction errors ($N=13,513$)
- **Calibrated Threshold:** `0.064806`
- **Score Statistics on Normal Val:**
  - Mean Error: `0.026189`
  - Std Error: `0.087801`
  - Median ($p_{50}$): `0.015779`
  - 95th Percentile ($p_{95}$): `0.064806`
  - 99th Percentile ($p_{99}$): `0.165632`
  - Max Error: `5.701713`

## 9. Performance Evaluation

### 9.1 Validation Split Metrics ($N=15,103$)
- **AUROC:** `0.923537`
- **AUPRC:** `0.709793`
- **Accuracy:** `0.928297` (92.83%)
- **Precision:** `0.636559` (63.66%)
- **Recall / Sensitivity:** `0.744186` (74.42%)
- **Specificity:** `0.949974` (95.00%)
- **F1-Score:** `0.686178` (68.62%)
- **Balanced Accuracy:** `0.847080` (84.71%)
- **Matthews Correlation Coefficient (MCC):** `0.648492`
- **False Positive Rate (FPR):** `0.050026` (5.00%)
- **FP per 1,000 Normal Samples:** `50.0`
- **Validation Confusion Matrix:**
  - True Negative ($TN$): `12,837`
  - False Positive ($FP$): `676`
  - False Negative ($FN$): `407`
  - True Positive ($TP$): `1,184`

### 9.2 Held-Out Test Split Metrics ($N=15,104$)
- **AUROC:** `0.921558` (92.16%)
- **AUPRC:** `0.696575` (69.66%)
- **Accuracy:** `0.923001` (92.30%)
- **Precision:** `0.612632` (61.26%)
- **Recall / Sensitivity:** `0.731615` (73.16%)
- **Specificity:** `0.945534` (94.55%)
- **F1-Score:** `0.666858` (66.69%)
- **Balanced Accuracy:** `0.838575` (83.86%)
- **Matthews Correlation Coefficient (MCC):** `0.626853`
- **False Positive Rate (FPR):** `0.054466` (5.45%)
- **FP per 1,000 Normal Samples:** `54.5`
- **Test Confusion Matrix:**
  - True Negative ($TN$): `12,777`
  - False Positive ($FP$): `736`
  - False Negative ($FN$): `427`
  - True Positive ($TP$): `1,164`

## 10. Computational Efficiency & Latency
- **Mean Latency per Sample:** `0.240 µs` (`0.000240 ms` — **Fastest Inference Speed**)
- **Median Latency ($p_{50}$):** `0.247 µs`
- **95th Percentile Latency ($p_{95}$):** `0.261 µs`
- **99th Percentile Latency ($p_{99}$):** `0.261 µs`
- **Batch Throughput:** `4,916,667 events/second` (**Top Throughput across all models**)

## 11. Visualizations
- `reports/figures/optimizer_convergence.png`: Purple curve showing COA convergence to 0.72598.
- `reports/figures/roc_curves_comparison.png`: Purple trace showing 0.9216 AUROC.
- `reports/figures/pr_curves_comparison.png`: Purple trace showing 0.6966 AUPRC.
- `reports/figures/confusion_matrices_all.png`: Subplot 4 showing COA confusion matrix.

## 12. Reproduction Command
```bash
python3 training/run_experiments.py --only coa_ae --n-agents 6 --n-iterations 5 --seed 42 --threshold-percentile 95.0
```

## 13. Scientific Interpretation
In accordance with unbiased empirical evaluation, COA did not yield superior classification metrics compared to GWO or the Baseline (AUROC 0.9216 vs. 0.9466 for GWO). However, COA converged toward an exceptionally compact topology (`32 -> 64 -> 32 -> 12`, 9,280 parameters) with batch size 128. This model achieves the highest inference throughput (4.92 million events/second) and lowest per-sample latency (0.24 microseconds). It is well-suited for edge devices and ultra-low-power embedded holter monitors where inference latency and memory footprint take precedence over marginal gains in discrimination.
