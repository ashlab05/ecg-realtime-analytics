#!/usr/bin/env python3
"""Generate publication-quality visualizations for ECG Anomaly Detection.

Produces figures in reports/figures/:
1. Training & Convergence:
   - loss_curves_comparison.png: Training & validation loss vs epoch for all models
   - optimizer_convergence.png: Best validation AUPRC vs iteration for PSO, GWO, COA
2. Classification Performance:
   - roc_curves_comparison.png: Multi-model ROC curves with AUROC
   - pr_curves_comparison.png: Multi-model Precision-Recall curves with AUPRC
   - confusion_matrices_all.png: Side-by-side 2x2 confusion matrices
   - metrics_bar_comparison.png: Key metrics bar comparison (AUROC, AUPRC, F1, Sens, Spec, MCC)
   - val_vs_test_comparison.png: Validation vs test generalization comparison
   - fpr_vs_recall_comparison.png: Trade-off between FPR and Abnormal Recall
3. Anomaly Detection & Explainability:
   - reconstruction_error_distribution.png: Normal vs Abnormal reconstruction error density & thresholds
   - representative_feature_reconstructions.png: Original vs Reconstructed 32-dim feature profiles
   - feature_level_error_anomalies.png: Squared error per feature for detected anomalies
   - top_contributing_features.png: Bar chart of top anomalous features
   - threshold_sensitivity.png: F1, Precision, and Recall vs Threshold Percentile
4. Robustness & Streaming:
   - noise_robustness_analysis.png: Performance degradation under Gaussian noise
   - distribution_drift_analysis.png: Sensitivity to simulated feature mean/variance shift
   - streaming_throughput_latency.png: Inference latency and throughput scaling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_curve, roc_curve

from ecg_core.features import FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES, N_FEATURES
from ecg_core.labels import NORMAL_CLASS
from ecg_core.preprocessing import load_dataset, make_split
from ecg_core.models.autoencoder import ECGAutoencoder

# Styling constants
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
TEXT_MAIN = "#e6edf3"
TEXT_DIM  = "#8b949e"
TEAL      = "#00d4aa"
ORANGE    = "#ff6b35"
PURPLE    = "#8b5cf6"
BLUE      = "#38bdf8"
RED       = "#f87171"
GREEN     = "#34d399"
GOLD      = "#fbbf24"

MODEL_COLORS = {
    "baseline_ae": BLUE,
    "pso_ae": ORANGE,
    "gwo_ae": GREEN,
    "coa_ae": PURPLE,
}

MODEL_LABELS = {
    "baseline_ae": "Baseline AE (Adam)",
    "pso_ae": "PSO-tuned AE",
    "gwo_ae": "GWO-tuned AE",
    "coa_ae": "COA-tuned AE",
}

OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT_MAIN,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "text.color": TEXT_MAIN,
        "grid.color": BORDER,
        "grid.alpha": 0.4,
        "legend.facecolor": CARD_BG,
        "legend.edgecolor": BORDER,
        "font.family": "sans-serif",
        "font.size": 10,
    })


def load_artifacts_data():
    models = ["baseline_ae", "pso_ae", "gwo_ae", "coa_ae"]
    data = {}
    for m in models:
        runs = sorted(Path(f"artifacts/experiments/{m}").glob("run_*"))
        p_run = runs[-1] if runs else Path(f"artifacts/experiments/{m}/latest")

        if p_run and (p_run / "metrics.json").exists():
            metrics = json.loads((p_run / "metrics.json").read_text())
            epoch_df = pd.read_csv(p_run / "epoch_history.csv")
            test_preds = pd.read_csv(p_run / "test_predictions.csv")
            val_preds = pd.read_csv(p_run / "val_predictions.csv")
            search_hist = None
            if (p_run / "search_history.json").exists():
                search_hist = json.loads((p_run / "search_history.json").read_text())

            data[m] = {
                "metrics": metrics,
                "epoch_df": epoch_df,
                "test_preds": test_preds,
                "val_preds": val_preds,
                "search_hist": search_hist,
                "dir": p_run,
            }
    return data


# 1. Training & Convergence Plots
def plot_loss_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor(DARK_BG)

    for m, info in data.items():
        c = MODEL_COLORS[m]
        lbl = MODEL_LABELS[m]
        df = info["epoch_df"]
        axes[0].plot(df["epoch"], df["train_loss"], color=c, label=lbl, lw=1.8, alpha=0.9)
        axes[1].plot(df["epoch"], df["val_loss"], color=c, label=lbl, lw=1.8, alpha=0.9)

    axes[0].set_title("Training Reconstruction Loss vs. Epoch", color=TEXT_MAIN, fontweight="bold")
    axes[0].set_xlabel("Epoch", color=TEXT_MAIN)
    axes[0].set_ylabel("Mean Squared Error (MSE)", color=TEXT_MAIN)
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].set_title("Validation Normal Reconstruction Loss vs. Epoch", color=TEXT_MAIN, fontweight="bold")
    axes[1].set_xlabel("Epoch", color=TEXT_MAIN)
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_curves_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "loss_curves_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: loss_curves_comparison.png / .pdf")


def plot_optimizer_convergence(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)

    for m in ["pso_ae", "gwo_ae", "coa_ae"]:
        if m in data and data[m]["search_hist"]:
            sh = data[m]["search_hist"]
            iters = [it["iteration"] for it in sh["iteration_history"]]
            best_fit = [it["best_fitness"] for it in sh["iteration_history"]]
            c = MODEL_COLORS[m]
            lbl = f"{MODEL_LABELS[m]} (Best: {sh['best_fitness']:.4f})"
            ax.plot(iters, best_fit, marker="o", lw=2.2, color=c, label=lbl)

    # Plot baseline reference line
    if "baseline_ae" in data:
        base_val_auprc = data["baseline_ae"]["metrics"]["validation_metrics"]["auprc"]
        ax.axhline(base_val_auprc, color=BLUE, ls="--", lw=1.5, label=f"Baseline AE (AUPRC: {base_val_auprc:.4f})")

    ax.set_title("Metaheuristic Optimization Convergence (Fitness: Val AUPRC)", color=TEXT_MAIN, fontweight="bold")
    ax.set_xlabel("Outer-Loop Optimization Iteration", color=TEXT_MAIN)
    ax.set_ylabel("Validation AUPRC", color=TEXT_MAIN)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "optimizer_convergence.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "optimizer_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: optimizer_convergence.png / .pdf")


# 2. Classification Performance Plots
def plot_roc_pr_curves(data):
    # ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_roc.patch.set_facecolor(DARK_BG)

    # PR Curves
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    fig_pr.patch.set_facecolor(DARK_BG)

    for m, info in data.items():
        c = MODEL_COLORS[m]
        df = info["test_preds"]
        y_true = df["binary_label"]
        scores = df["reconstruction_error"]

        # ROC
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = info["metrics"]["test_metrics"]["auroc"]
        ax_roc.plot(fpr, tpr, color=c, lw=2.0, label=f"{MODEL_LABELS[m]} (AUROC = {auroc:.4f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_true, scores)
        auprc = info["metrics"]["test_metrics"]["auprc"]
        ax_pr.plot(rec, prec, color=c, lw=2.0, label=f"{MODEL_LABELS[m]} (AUPRC = {auprc:.4f})")

    # Finalize ROC
    ax_roc.plot([0, 1], [0, 1], color=TEXT_DIM, ls=":", lw=1.2, label="Random Guess")
    ax_roc.set_title("ROC Curves Comparison (Held-out Test Set)", color=TEXT_MAIN, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (FPR)", color=TEXT_MAIN)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", color=TEXT_MAIN)
    ax_roc.set_xlim(-0.01, 1.01)
    ax_roc.set_ylim(-0.01, 1.01)
    ax_roc.grid(True)
    ax_roc.legend(loc="lower right", fontsize=9)
    fig_roc.tight_layout()
    fig_roc.savefig(OUT_DIR / "roc_curves_comparison.png", dpi=200, bbox_inches="tight")
    fig_roc.savefig(OUT_DIR / "roc_curves_comparison.pdf", bbox_inches="tight")
    plt.close(fig_roc)
    print("Saved: roc_curves_comparison.png / .pdf")

    # Finalize PR
    # Baseline prevalence line
    prev = data["baseline_ae"]["test_preds"]["binary_label"].mean()
    ax_pr.axhline(prev, color=TEXT_DIM, ls=":", lw=1.2, label=f"Prevalence ({prev:.3f})")
    ax_pr.set_title("Precision-Recall Curves Comparison (Held-out Test Set)", color=TEXT_MAIN, fontweight="bold")
    ax_pr.set_xlabel("Recall (Sensitivity)", color=TEXT_MAIN)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", color=TEXT_MAIN)
    ax_pr.set_xlim(-0.01, 1.01)
    ax_pr.set_ylim(-0.01, 1.01)
    ax_pr.grid(True)
    ax_pr.legend(loc="lower left", fontsize=9)
    fig_pr.tight_layout()
    fig_pr.savefig(OUT_DIR / "pr_curves_comparison.png", dpi=200, bbox_inches="tight")
    fig_pr.savefig(OUT_DIR / "pr_curves_comparison.pdf", bbox_inches="tight")
    plt.close(fig_pr)
    print("Saved: pr_curves_comparison.png / .pdf")


def plot_confusion_matrices(data):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    fig.patch.set_facecolor(DARK_BG)

    models = ["baseline_ae", "pso_ae", "gwo_ae", "coa_ae"]
    for idx, m in enumerate(models):
        ax = axes[idx]
        if m not in data:
            ax.axis("off")
            continue
        tm = data[m]["metrics"]["test_metrics"]
        cm = tm["confusion_matrix"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        mat = np.array([[tn, fp], [fn, tp]])

        # Display matrix
        cax = ax.imshow(mat, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{MODEL_LABELS[m]}\nF1={tm['f1_score']:.4f} | Rec={tm['recall']:.4f}",
                     color=TEXT_MAIN, fontweight="bold", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Norm (0)", "Pred Abn (1)"], color=TEXT_DIM, fontsize=9)
        ax.set_yticklabels(["True Norm (0)", "True Abn (1)"], color=TEXT_DIM, fontsize=9)

        # Annotations
        thresh = mat.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                pct = val / mat.sum() * 100.0
                text_col = "white" if val > thresh else "black"
                ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center",
                        color=text_col, fontweight="bold", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices_all.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "confusion_matrices_all.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: confusion_matrices_all.png / .pdf")


def plot_metrics_bars(data):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    models = ["baseline_ae", "pso_ae", "gwo_ae", "coa_ae"]
    metric_keys = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("f1_score", "F1-Score"),
        ("recall", "Sensitivity"),
        ("specificity", "Specificity"),
        ("mcc", "MCC"),
    ]

    n_models = len(models)
    n_metrics = len(metric_keys)
    x = np.arange(n_metrics)
    width = 0.18

    for i, m in enumerate(models):
        if m in data:
            vals = [data[m]["metrics"]["test_metrics"][k] for k, _ in metric_keys]
            ax.bar(x + (i - 1.5) * width, vals, width, label=MODEL_LABELS[m], color=MODEL_COLORS[m], alpha=0.9)

    ax.set_title("Test Set Anomaly Detection Metrics Comparison", color=TEXT_MAIN, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metric_keys], color=TEXT_MAIN, fontweight="bold")
    ax.set_ylabel("Metric Value", color=TEXT_MAIN)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "metrics_bar_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "metrics_bar_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: metrics_bar_comparison.png / .pdf")


def plot_val_vs_test_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(DARK_BG)

    models = ["baseline_ae", "pso_ae", "gwo_ae", "coa_ae"]
    x = np.arange(len(models))
    width = 0.35

    # AUROC
    val_auroc = [data[m]["metrics"]["validation_metrics"]["auroc"] for m in models]
    test_auroc = [data[m]["metrics"]["test_metrics"]["auroc"] for m in models]
    axes[0].bar(x - width/2, val_auroc, width, label="Validation AUROC", color=BLUE, alpha=0.85)
    axes[0].bar(x + width/2, test_auroc, width, label="Test AUROC", color=TEAL, alpha=0.85)
    axes[0].set_title("AUROC: Validation vs. Test Split", color=TEXT_MAIN, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODEL_LABELS[m].split()[0] for m in models], color=TEXT_MAIN)
    axes[0].set_ylim(0.88, 0.98)
    axes[0].grid(True, axis="y")
    axes[0].legend(loc="lower right")

    # AUPRC
    val_auprc = [data[m]["metrics"]["validation_metrics"]["auprc"] for m in models]
    test_auprc = [data[m]["metrics"]["test_metrics"]["auprc"] for m in models]
    axes[1].bar(x - width/2, val_auprc, width, label="Validation AUPRC", color=ORANGE, alpha=0.85)
    axes[1].bar(x + width/2, test_auprc, width, label="Test AUPRC", color=GOLD, alpha=0.85)
    axes[1].set_title("AUPRC: Validation vs. Test Split", color=TEXT_MAIN, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_LABELS[m].split()[0] for m in models], color=TEXT_MAIN)
    axes[1].set_ylim(0.65, 0.82)
    axes[1].grid(True, axis="y")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "val_vs_test_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "val_vs_test_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: val_vs_test_comparison.png / .pdf")


def plot_fpr_vs_recall(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)

    for m, info in data.items():
        tm = info["metrics"]["test_metrics"]
        fpr_pct = tm["false_positive_rate"] * 100.0
        rec_pct = tm["recall"] * 100.0
        c = MODEL_COLORS[m]
        lbl = MODEL_LABELS[m]
        ax.scatter([fpr_pct], [rec_pct], color=c, s=140, zorder=5, label=f"{lbl} ({fpr_pct:.2f}%, {rec_pct:.2f}%)")
        ax.annotate(lbl.split()[0], (fpr_pct + 0.08, rec_pct - 0.2), color=c, fontweight="bold", fontsize=10)

    ax.set_title("False Positive Rate vs. Abnormal Recall (Sensitivity)", color=TEXT_MAIN, fontweight="bold")
    ax.set_xlabel("False Positive Rate (%) — (Lower is Better)", color=TEXT_MAIN)
    ax.set_ylabel("Abnormal Recall (%) — (Higher is Better)", color=TEXT_MAIN)
    ax.grid(True)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fpr_vs_recall_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fpr_vs_recall_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: fpr_vs_recall_comparison.png / .pdf")


# 3. Anomaly Detection & Explainability
def plot_error_distribution(data):
    # Plot distribution for top performing model: GWO-AE
    best_model = "gwo_ae" if "gwo_ae" in data else "baseline_ae"
    preds = data[best_model]["test_preds"]
    thresh = data[best_model]["metrics"]["threshold"]

    norm_errors = preds[preds["binary_label"] == 0]["reconstruction_error"]
    abn_errors = preds[preds["binary_label"] == 1]["reconstruction_error"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)

    # Truncate upper tail for visual clarity
    clip_val = np.percentile(abn_errors, 98)
    norm_clip = norm_errors[norm_errors < clip_val]
    abn_clip = abn_errors[abn_errors < clip_val]

    bins = np.linspace(0, clip_val, 70)
    ax.hist(norm_clip, bins=bins, density=True, alpha=0.65, color=GREEN, label=f"Normal Beats (N = {len(norm_errors):,})")
    ax.hist(abn_clip, bins=bins, density=True, alpha=0.65, color=RED, label=f"Abnormal Beats (N = {len(abn_errors):,})")

    ax.axvline(thresh, color=GOLD, ls="--", lw=2.2, label=f"Calibrated Threshold ({thresh:.4f})")

    ax.set_title(f"Reconstruction Error Distribution: Normal vs. Abnormal ({MODEL_LABELS[best_model]})",
                 color=TEXT_MAIN, fontweight="bold")
    ax.set_xlabel("Reconstruction Error (Mean Squared Error across 32 Features)", color=TEXT_MAIN)
    ax.set_ylabel("Probability Density", color=TEXT_MAIN)
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "reconstruction_error_distribution.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "reconstruction_error_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: reconstruction_error_distribution.png / .pdf")


def plot_representative_reconstructions(data):
    # Representative normal beat and abnormal beat feature reconstruction profiles
    best_model = "gwo_ae" if "gwo_ae" in data else "baseline_ae"
    p_weights = data[best_model]["dir"] / "best_model.pt"
    p_meta = data[best_model]["dir"] / "config.json"
    meta = json.loads(p_meta.read_text())
    hp = meta["hyperparameters"]

    model = ECGAutoencoder(
        n_features=N_FEATURES,
        hidden_dims=tuple(hp["hidden_dims"]),
        latent_dim=hp["latent_dim"],
        dropout=hp["dropout"],
        noise_std=0.0,
    )
    import torch
    model.load_state_dict(torch.load(p_weights, map_location="cpu", weights_only=True))
    model.eval()

    df = load_dataset()
    split = make_split(df, seed=42)

    # Pick typical normal sample and severe VEB sample
    idx_norm = np.where(split.y_test == "N")[0][10]
    idx_abn = np.where(split.y_test == "VEB")[0][5]

    x_norm = split.X_test_scaled[idx_norm:idx_norm+1]
    x_abn = split.X_test_scaled[idx_abn:idx_abn+1]

    recon_norm = model.reconstruct(x_norm)[0]
    recon_abn = model.reconstruct(x_abn)[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    feat_x = np.arange(N_FEATURES)

    # Normal sample
    axes[0].plot(feat_x, x_norm[0], color=TEAL, marker="o", label="Original Standardized Features", lw=1.8)
    axes[0].plot(feat_x, recon_norm, color=BLUE, marker="x", ls="--", label="Autoencoder Reconstruction", lw=1.8)
    axes[0].set_title(f"Representative Normal Beat Profile (Sample #{idx_norm}, Label: N) — MSE: {np.mean((x_norm[0] - recon_norm)**2):.4f}",
                     color=TEXT_MAIN, fontweight="bold")
    axes[0].set_ylabel("Standardized Value", color=TEXT_MAIN)
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # Abnormal sample
    axes[1].plot(feat_x, x_abn[0], color=RED, marker="o", label="Original Standardized Features", lw=1.8)
    axes[1].plot(feat_x, recon_abn, color=GOLD, marker="x", ls="--", label="Autoencoder Reconstruction", lw=1.8)
    axes[1].set_title(f"Representative Arrhythmic Beat Profile (Sample #{idx_abn}, Label: VEB) — MSE: {np.mean((x_abn[0] - recon_abn)**2):.4f}",
                     color=TEXT_MAIN, fontweight="bold")
    axes[1].set_ylabel("Standardized Value", color=TEXT_MAIN)
    axes[1].set_xlabel("Feature Index (0–15: Lead II, 16–31: Lead V)", color=TEXT_MAIN)
    axes[1].set_xticks(feat_x)
    axes[1].set_xticklabels([c.replace("0_", "II:").replace("1_", "V:") for c in FEATURE_COLUMNS],
                            rotation=90, fontsize=8)
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "representative_feature_reconstructions.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "representative_feature_reconstructions.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: representative_feature_reconstructions.png / .pdf")


def plot_feature_level_error_and_top_features(data):
    best_model = "gwo_ae" if "gwo_ae" in data else "baseline_ae"
    p_weights = data[best_model]["dir"] / "best_model.pt"
    p_meta = data[best_model]["dir"] / "config.json"
    meta = json.loads(p_meta.read_text())
    hp = meta["hyperparameters"]

    model = ECGAutoencoder(
        n_features=N_FEATURES,
        hidden_dims=tuple(hp["hidden_dims"]),
        latent_dim=hp["latent_dim"],
        dropout=hp["dropout"],
        noise_std=0.0,
    )
    import torch
    model.load_state_dict(torch.load(p_weights, map_location="cpu", weights_only=True))
    model.eval()

    df = load_dataset()
    split = make_split(df, seed=42)

    # Compute per-feature squared error across all test abnormal beats vs normal beats
    abnormal_mask = split.y_test != NORMAL_CLASS
    X_test_abn = split.X_test_scaled[abnormal_mask]
    X_test_norm = split.X_test_scaled[~abnormal_mask]

    recon_abn = model.reconstruct(X_test_abn)
    recon_norm = model.reconstruct(X_test_norm)

    feat_sq_err_abn = np.mean((X_test_abn - recon_abn) ** 2, axis=0)
    feat_sq_err_norm = np.mean((X_test_norm - recon_norm) ** 2, axis=0)

    # 1. Feature Level Error comparison
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    x = np.arange(N_FEATURES)
    width = 0.4
    ax.bar(x - width/2, feat_sq_err_norm, width, label="Normal Beats Mean Squared Error", color=TEAL, alpha=0.85)
    ax.bar(x + width/2, feat_sq_err_abn, width, label="Abnormal Beats Mean Squared Error", color=RED, alpha=0.85)

    ax.set_title("Mean Squared Reconstruction Error per Feature: Normal vs. Abnormal Beats", color=TEXT_MAIN, fontweight="bold")
    ax.set_ylabel("Mean Squared Error", color=TEXT_MAIN)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("0_", "II: ").replace("1_", "V: ") for c in FEATURE_COLUMNS],
                       rotation=90, fontsize=8)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_level_error_anomalies.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "feature_level_error_anomalies.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: feature_level_error_anomalies.png / .pdf")

    # 2. Top Contributing Features (Difference between abnormal and normal reconstruction error)
    delta_err = feat_sq_err_abn - feat_sq_err_norm
    top_indices = np.argsort(delta_err)[-10:][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)

    top_names = [FEATURE_DISPLAY_NAMES[FEATURE_COLUMNS[i]] for i in top_indices]
    top_vals = [delta_err[i] for i in top_indices]

    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals, color=ORANGE, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, color=TEXT_MAIN, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"Reconstruction Error Discrepancy ($\Delta$ MSE Abnormal - Normal)", color=TEXT_MAIN)
    ax.set_title("Top 10 Diagnostic Features Driving Anomaly Gating Decisions", color=TEXT_MAIN, fontweight="bold")
    ax.grid(True, axis="x")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_contributing_features.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "top_contributing_features.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: top_contributing_features.png / .pdf")


def plot_threshold_sensitivity(data):
    best_model = "gwo_ae" if "gwo_ae" in data else "baseline_ae"
    test_preds = data[best_model]["test_preds"]
    val_preds = data[best_model]["val_preds"]

    # Compute percentiles of normal validation errors
    val_norm_errs = val_preds[val_preds["binary_label"] == 0]["reconstruction_error"]
    y_test = test_preds["binary_label"].to_numpy()
    test_scores = test_preds["reconstruction_error"].to_numpy()

    percentiles = np.linspace(80, 99.5, 40)
    f1_list, prec_list, rec_list = [], [], []

    for p in percentiles:
        th = np.percentile(val_norm_errs, p)
        pred = (test_scores > th).astype(int)
        tp = np.sum((y_test == 1) & (pred == 1))
        fp = np.sum((y_test == 0) & (pred == 1))
        fn = np.sum((y_test == 1) & (pred == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)

    ax.plot(percentiles, f1_list, color=GOLD, lw=2.2, label="F1-Score")
    ax.plot(percentiles, prec_list, color=BLUE, lw=2.0, label="Precision")
    ax.plot(percentiles, rec_list, color=TEAL, lw=2.0, label="Recall (Sensitivity)")

    # Highlight p95
    th95_idx = np.argmin(np.abs(percentiles - 95.0))
    ax.axvline(95.0, color=RED, ls="--", lw=1.5, label=f"Calibrated Policy (p95: F1={f1_list[th95_idx]:.3f})")

    ax.set_title("Threshold Calibration Sensitivity Analysis (Normal Validation Percentile)",
                 color=TEXT_MAIN, fontweight="bold")
    ax.set_xlabel("Calibration Percentile on Normal Validation Errors (%)", color=TEXT_MAIN)
    ax.set_ylabel("Metric Value on Held-out Test Set", color=TEXT_MAIN)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "threshold_sensitivity.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "threshold_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: threshold_sensitivity.png / .pdf")


# 4. Robustness & Streaming Plots
def plot_robustness_and_streaming(data):
    best_model = "gwo_ae" if "gwo_ae" in data else "baseline_ae"
    p_weights = data[best_model]["dir"] / "best_model.pt"
    p_meta = data[best_model]["dir"] / "config.json"
    meta = json.loads(p_meta.read_text())
    hp = meta["hyperparameters"]
    threshold = meta["threshold_calibration"]["threshold_value"]

    model = ECGAutoencoder(
        n_features=N_FEATURES,
        hidden_dims=tuple(hp["hidden_dims"]),
        latent_dim=hp["latent_dim"],
        dropout=hp["dropout"],
        noise_std=0.0,
    )
    import torch
    model.load_state_dict(torch.load(p_weights, map_location="cpu", weights_only=True))
    model.eval()

    df = load_dataset()
    split = make_split(df, seed=42)
    X_test = split.X_test_scaled
    y_test = (split.y_test != NORMAL_CLASS).astype(int)

    # 1. Noise Robustness: Gaussian noise sigma in [0, 0.50]
    rng = np.random.default_rng(42)
    sigmas = np.linspace(0.0, 0.50, 11)
    auroc_noise, f1_noise = [], []
    from sklearn.metrics import roc_auc_score, f1_score

    for s in sigmas:
        noisy_X = X_test + rng.normal(0, s, size=X_test.shape)
        recon = model.reconstruct(noisy_X)
        err = np.mean((noisy_X - recon)**2, axis=1)
        auroc_noise.append(roc_auc_score(y_test, err))
        f1_noise.append(f1_score(y_test, err > threshold))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(sigmas, auroc_noise, marker="o", color=BLUE, lw=2.0, label="Test AUROC")
    ax.plot(sigmas, f1_noise, marker="s", color=ORANGE, lw=2.0, label="Test F1-Score")
    ax.set_title(f"Model Robustness Under Additive Gaussian Feature Noise ({MODEL_LABELS[best_model]})",
                 color=TEXT_MAIN, fontweight="bold")
    ax.set_xlabel(r"Noise Standard Deviation ($\sigma$)", color=TEXT_MAIN)
    ax.set_ylabel("Metric Value", color=TEXT_MAIN)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "noise_robustness_analysis.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "noise_robustness_analysis.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: noise_robustness_analysis.png / .pdf")

    # 2. Simulated Distribution Drift: shift mean of Lead II / Lead V features
    shifts = np.linspace(0.0, 1.0, 11)
    auroc_drift, fpr_drift = [], []
    for shift in shifts:
        drifted_X = X_test.copy()
        drifted_X[:, :16] += shift  # Mean drift on Lead II
        recon = model.reconstruct(drifted_X)
        err = np.mean((drifted_X - recon)**2, axis=1)
        auroc_drift.append(roc_auc_score(y_test, err))
        fp = np.sum((y_test == 0) & (err > threshold))
        tn = np.sum((y_test == 0) & (err <= threshold))
        fpr_drift.append(fp / (fp + tn))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(shifts, auroc_drift, marker="o", color=TEAL, lw=2.0, label="Test AUROC")
    ax.plot(shifts, [fpr * 100 for fpr in fpr_drift], marker="^", color=RED, lw=2.0, label="False Positive Rate (%)")
    ax.set_title("Performance & False Alarms Under Simulated Lead II Distribution Drift",
                 color=TEXT_MAIN, fontweight="bold")
    ax.set_xlabel(r"Standardized Mean Shift Magnitude ($\Delta \mu$)", color=TEXT_MAIN)
    ax.set_ylabel("Value", color=TEXT_MAIN)
    ax.grid(True)
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distribution_drift_analysis.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "distribution_drift_analysis.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: distribution_drift_analysis.png / .pdf")

    # 3. Streaming Throughput and Latency scaling across batch sizes
    batch_sizes = [1, 16, 64, 256, 1024, 4096]
    latencies_us = []
    throughputs = []
    import time
    for b in batch_sizes:
        sub_X = X_test[:min(b, len(X_test))]
        # warmup
        model.reconstruct(sub_X)
        t0 = time.perf_counter()
        for _ in range(20):
            model.reconstruct(sub_X)
        dt = (time.perf_counter() - t0) / 20.0
        lat_per_event_us = (dt / len(sub_X)) * 1_000_000.0  # microseconds
        thr_eps = len(sub_X) / dt if dt > 0 else 0
        latencies_us.append(lat_per_event_us)
        throughputs.append(thr_eps)

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor(DARK_BG)
    c1 = TEAL
    c2 = GOLD
    ax1.plot(batch_sizes, latencies_us, marker="o", color=c1, lw=2.2, label=r"Latency per Beat ($\mu$s)")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Batch Size (samples)", color=TEXT_MAIN)
    ax1.set_ylabel(r"Inference Latency per Sample ($\mu$s)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(batch_sizes, [t / 1000.0 for t in throughputs], marker="s", color=c2, lw=2.2, ls="--", label="Throughput (k events/s)")
    ax2.set_ylabel("Inference Throughput (thousand events / sec)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    ax1.set_title(f"Inference Latency & Throughput Scaling ({MODEL_LABELS[best_model]})", color=TEXT_MAIN, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "streaming_throughput_latency.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "streaming_throughput_latency.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved: streaming_throughput_latency.png / .pdf")


def main():
    apply_dark_style()
    data = load_artifacts_data()
    print(f"Loaded experiment data for: {list(data.keys())}")

    print("\n--- Generating Training & Convergence Visualizations ---")
    plot_loss_curves(data)
    plot_optimizer_convergence(data)

    print("\n--- Generating Classification Performance Visualizations ---")
    plot_roc_pr_curves(data)
    plot_confusion_matrices(data)
    plot_metrics_bars(data)
    plot_val_vs_test_comparison(data)
    plot_fpr_vs_recall(data)

    print("\n--- Generating Anomaly Detection & Explainability Visualizations ---")
    plot_error_distribution(data)
    plot_representative_reconstructions(data)
    plot_feature_level_error_and_top_features(data)
    plot_threshold_sensitivity(data)

    print("\n--- Generating Robustness & Streaming Visualizations ---")
    plot_robustness_and_streaming(data)

    print(f"\nAll publication figures successfully generated in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
