# Comprehensive Experiment Comparison: Autoencoder Optimization Methods

**Project:** ECG Real-Time Arrhythmia Analytics  
**Date:** 2026-10-24  
**Split Protocol:** MIT-BIH Arrhythmia Database (70% Train, 15% Validation, 15% Held-out Test; `seed=42`)  
**Task:** Binary Anomaly Detection (Normal $N=0$ vs. Arrhythmic Abnormal $A=1$)  

---

## 1. Consolidated Performance Table

The table below presents the empirical evaluation across all four experimental configurations evaluated on the identical held-out test split ($N=15,104$ beats: 13,513 Normal, 1,591 Abnormal).

| Metric / Dimension | Baseline AE + Adam | PSO-Tuned AE + Adam | GWO-Tuned AE + Adam | COA-Tuned AE + Adam |
| :--- | :---: | :---: | :---: | :---: |
| **Outer Optimizer** | None (Default) | Particle Swarm (PSO) | Grey Wolf (GWO) | Coati (COA) |
| **Inner Optimizer** | Adam | Adam | Adam | Adam |
| **Model Type** | Standard AE | Denoising AE | Denoising AE | Denoising AE |
| **Hidden Layers** | `128 → 64` | `192 → 96` | `192 → 96` | `64 → 32` |
| **Latent Dimension** | 16 | 24 | **32** | 12 |
| **Trainable Parameters** | 26,080 | 54,944 | 57,280 | **9,280** |
| **Learning Rate** | `1.00e-3` | `4.16e-3` | `1.26e-3` | `1.38e-3` |
| **Weight Decay** | `0.0` | `2.53e-5` | `2.20e-6` | `4.67e-5` |
| **Dropout Rate** | 0.100 | 0.103 | 0.034 | 0.033 |
| **Denoising Noise ($\sigma$)** | 0.000 | 0.181 | 0.116 | 0.112 |
| **Batch Size** | 256 | 512 | 512 | 128 |
| **Calibrated Threshold ($p_{95}$)** | `0.05783` | `0.05357` | `0.01631` | `0.06481` |
| **Val AUPRC (Fitness)** | `0.7505` | `0.7385` | **0.7726** | `0.7098` |
| **Val AUROC** | `0.9430` | **0.9492** | `0.9483` | `0.9235` |
| **Test AUROC** | `0.9404` | `0.9451` | **0.9466** | `0.9216` |
| **Test AUPRC** | `0.7346` | `0.7289` | **0.7612** | `0.6966` |
| **Test F1-Score** | `0.6973` | `0.6903` | **0.6976** | `0.6669` |
| **Test Precision** | **0.6394** | `0.6294` | `0.6276` | `0.6126` |
| **Test Sensitivity (Recall)** | `0.7668` | `0.7643` | **0.7850** | `0.7316` |
| **Test Specificity** | **0.9491** | `0.9470` | `0.9452` | `0.9455` |
| **Test Overall Accuracy** | **0.9299** | `0.9278` | `0.9283` | `0.9230` |
| **Test Balanced Accuracy** | `0.8580` | `0.8557` | **0.8651** | `0.8386` |
| **Test MCC** | `0.6615` | `0.6538` | **0.6628** | `0.6269` |
| **False Positive Rate (FPR)** | **0.0509** (5.09%) | `0.0530` (5.30%) | `0.0548` (5.48%) | `0.0545` (5.45%) |
| **FP / 1,000 Normal Beats** | **50.9** | `53.0` | `54.8` | `54.5` |
| **Inference Latency (Mean)** | `0.357 µs` | `0.573 µs` | `0.551 µs` | **0.240 µs** |
| **Inference Throughput** | `2.97M ev/s` | `1.92M ev/s` | `1.90M ev/s` | **4.92M ev/s** |
| **Optimization Budget** | 0 evals | 36 evals | 36 evals | 36 evals |
| **Optimization Duration** | `0.0 s` | `108.7 s` | `139.6 s` | `118.0 s` |
| **Full Model Train Time** | `11.8 s` (50 ep) | **7.7 s** (27 ep) | `14.3 s` (50 ep) | `12.4 s` (50 ep) |

---

## 2. Test Confusion Matrices Breakdown

| Model / Metric | True Negatives ($TN$) | False Positives ($FP$) | False Negatives ($FN$) | True Positives ($TP$) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline AE** | **12,825** | **688** | 371 | 1,220 |
| **PSO-Tuned AE** | 12,797 | 716 | 375 | 1,216 |
| **GWO-Tuned AE** | 12,772 | 741 | **342** | **1,249** |
| **COA-Tuned AE** | 12,777 | 736 | 427 | 1,164 |

---

## 3. Comparative Findings & Key Trade-Offs

### 3.1 Discrimination Power: GWO-Tuned AE as the Leading Architecture
- **GWO-Tuned AE** attained the **highest Test AUROC (0.9466)** and **highest Test AUPRC (0.7612)**, yielding a +2.67% improvement in average precision over the un-tuned Baseline.
- Crucially for clinical applications, GWO achieved the **highest abnormal beat sensitivity / recall of 78.50%** (detecting 1,249 of the 1,591 test anomalies, compared to 1,220 for Baseline and 1,164 for COA).
- The key architectural factor identified by GWO was expanding the latent bottleneck to 32 dimensions paired with input denoising ($\sigma = 0.1157$). This prevented information bottlenecking while forcing the network to denoise and accurately reconstruct clean normal morphologies.

### 3.2 Computational Throughput: COA-Tuned AE as the Edge Inference Champion
- **COA-Tuned AE** selected a significantly more compact parameterization (`32 → 64 → 32 → 12`, 9,280 parameters).
- While this smaller capacity led to a lower test AUROC (0.9216) and AUPRC (0.6966), it achieved an extraordinary inference throughput of **4.92 million heartbeats per second** and a latency of only **0.240 microseconds per beat**.
- For real-time streaming pipelines or battery-constrained ambulatory Holter monitors, COA offers a viable lightweight deployment option.

### 3.3 Convergence and Regularization: PSO-Tuned AE
- **PSO-Tuned AE** demonstrated rapid training convergence. By discovering effective weight decay ($2.53 \times 10^{-5}$) and dropout ($0.1026$), its early stopping mechanism halted full training at epoch 27 (requiring just 7.73 seconds of compute).
- It achieved a solid test AUROC of 0.9451, improving upon the baseline while avoiding over-parameterization.

---

## 4. Methodological Rigor & Fairness Verification
1. **Identical Splitting:** All four models were trained, calibrated, and tested on the exact same indices of the stratified MIT-BIH dataset (`seed=42`).
2. **Identical Search Budget:** All three outer-loop metaheuristics evaluated exactly 36 candidate models (6 initial + 30 iterative evaluations) under the same 15-epoch early-stopped fitness harness.
3. **Unbiased Selection:** Hyperparameters were chosen solely on the basis of **Validation AUPRC** without touching or peeking at test split labels.
4. **Calibrated Decision Boundary:** Thresholds were calibrated strictly as the 95th percentile of normal validation reconstruction errors before test evaluation.
