#!/usr/bin/env python3
"""Master execution script for Autoencoder Anomaly Detection Experiments.

Runs and compares:
  1. Baseline AE + Adam
  2. PSO-tuned AE + Adam
  3. GWO-tuned AE + Adam
  4. COA-tuned AE + Adam

Saves reproducible, publication-ready artifacts into:
  artifacts/experiments/<model_name>/<run_id>/
and syncs summary JSONs to:
  logs/training/experiments/<model_name>.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_core.config import settings
from ecg_core.features import FEATURE_COLUMNS, N_FEATURES
from ecg_core.labels import NORMAL_CLASS
from ecg_core.metrics import compute_anomaly_metrics
from ecg_core.models.autoencoder import ECGAutoencoder
from ecg_core.preprocessing import load_dataset, make_split, save_artifacts, PreprocessingArtifacts
from ecg_core.training_utils import EarlyStopping, seed_everything
from training.metaheuristics.coa import run_coa
from training.metaheuristics.evaluator import evaluate_candidate
from training.metaheuristics.gwo import run_gwo
from training.metaheuristics.pso import run_pso
from training.metaheuristics.search_space import (
    Hyperparameters,
    decode_vector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s]: %(message)s")
logger = logging.getLogger("run_experiments")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "uncommitted_or_non_git"


def get_system_metadata() -> Dict:
    return {
        "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": "cpu",
    }


def measure_inference_performance(model: ECGAutoencoder, X: np.ndarray, repeats: int = 5) -> Tuple[Dict, float]:
    """Measure offline inference latency per sample and overall throughput."""
    model.eval()
    latencies = []
    throughputs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        recon = model.reconstruct(X)
        _ = np.mean((X - recon) ** 2, axis=1)
        dt = time.perf_counter() - t0
        latencies.append((dt / len(X)) * 1000.0)  # ms per sample
        throughputs.append(len(X) / dt if dt > 0 else 0.0)
    arr = np.asarray(latencies, dtype=float)
    stats = {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
    return stats, float(max(throughputs))


def train_full_model(
    hp: Hyperparameters,
    X_train_normal: np.ndarray,
    X_val_normal: np.ndarray,
    max_epochs: int = 50,
    patience: int = 10,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[ECGAutoencoder, pd.DataFrame, Dict]:
    """Train the final Autoencoder with full epoch budget on normal beats."""
    seed_everything(seed)
    n_features = X_train_normal.shape[1]
    model = ECGAutoencoder(
        n_features=n_features,
        hidden_dims=hp.hidden_dims,
        latent_dim=hp.latent_dim,
        dropout=hp.dropout,
        noise_std=hp.noise_std,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
    )
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.as_tensor(X_train_normal, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True, drop_last=False)
    val_tensor = torch.as_tensor(X_val_normal, dtype=torch.float32).to(device)

    stopper = EarlyStopping(patience=patience)
    history = []
    best_loss = float("inf")
    best_epoch = 1
    best_weights = None

    start_train_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        for (batch,) in train_dl:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor)
            val_loss = float(criterion(val_recon, val_tensor).item())

        epoch_duration = time.perf_counter() - t_epoch_start

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_duration_s": epoch_duration,
            "total_elapsed_s": time.perf_counter() - start_train_time,
        })

        if epoch % 5 == 0 or epoch == 1:
            logger.info("Epoch %02d/%02d | train_loss=%.6f | val_loss=%.6f (best=%.6f at ep %d)",
                        epoch, max_epochs, train_loss, val_loss, best_loss, best_epoch)

        if stopper.step(val_loss):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    total_train_time = time.perf_counter() - start_train_time

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    train_meta = {
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_train_loss": float(history[best_epoch - 1]["train_loss"]),
        "best_val_loss": float(best_loss),
        "total_train_time_s": total_train_time,
        "early_stopped": len(history) < max_epochs,
    }
    return model, pd.DataFrame(history), train_meta


def execute_experiment(
    experiment_name: str,
    optimizer_type: str,  # "none", "pso", "gwo", "coa"
    split_strategy: str = "stratified",
    n_agents: int = 6,
    n_iterations: int = 5,
    seed: int = 42,
    threshold_percentile: float = 95.0,
) -> Dict:
    """Execute a complete experiment pipeline with logging and artifact creation."""
    logger.info("=================================================================")
    logger.info("STARTING EXPERIMENT: %s (Optimizer: %s)", experiment_name, optimizer_type)
    logger.info("=================================================================")

    start_exp_time = time.perf_counter()
    seed_everything(seed)

    df = load_dataset()
    split = make_split(df, strategy=split_strategy, seed=seed)

    # Normal training samples for AE
    X_train_normal = split.X_train_scaled[split.y_train == NORMAL_CLASS]
    X_val_normal = split.X_val_scaled[split.y_val == NORMAL_CLASS]

    # Binary labels for validation and test: Normal=0, Abnormal=1
    y_val_binary = (split.y_val != NORMAL_CLASS).astype(int)
    y_test_binary = (split.y_test != NORMAL_CLASS).astype(int)

    # 1. Hyperparameter Selection
    search_history = None
    opt_time_s = 0.0

    if optimizer_type == "none":
        # Baseline default configuration
        hp = Hyperparameters(
            latent_dim=16,
            hidden_dims=(128, 64),
            lr=1e-3,
            weight_decay=0.0,
            dropout=0.10,
            noise_std=0.0,
            batch_size=256,
        )
        logger.info("Using baseline hyperparameters: %s", hp.to_dict())
    else:
        # Define evaluation closure for outer-loop metaheuristic
        def candidate_eval(cand_hp: Hyperparameters):
            return evaluate_candidate(
                hp=cand_hp,
                X_train_normal=X_train_normal,
                X_val=split.X_val_scaled,
                y_val_binary=y_val_binary,
                max_epochs=15,
                patience=3,
                device="cpu",
                seed=seed,
            )

        opt_start = time.perf_counter()
        if optimizer_type == "pso":
            opt_result = run_pso(candidate_eval, n_particles=n_agents, n_iterations=n_iterations, seed=seed)
        elif optimizer_type == "gwo":
            opt_result = run_gwo(candidate_eval, n_wolves=n_agents, n_iterations=n_iterations, seed=seed)
        elif optimizer_type == "coa":
            opt_result = run_coa(candidate_eval, n_coatis=n_agents, n_iterations=n_iterations, seed=seed)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        opt_time_s = time.perf_counter() - opt_start
        search_history = opt_result
        best_hp_dict = opt_result["best_hyperparameters"]
        hp = Hyperparameters(
            latent_dim=best_hp_dict["latent_dim"],
            hidden_dims=tuple(best_hp_dict["hidden_dims"]),
            lr=best_hp_dict["lr"],
            weight_decay=best_hp_dict["weight_decay"],
            dropout=best_hp_dict["dropout"],
            noise_std=best_hp_dict["noise_std"],
            batch_size=best_hp_dict["batch_size"],
        )
        logger.info("Optimizer %s selected best hyperparameters (fitness=%.5f): %s",
                    optimizer_type, opt_result["best_fitness"], hp.to_dict())

    # 2. Train Full Model on Normal training samples
    model, epoch_df, train_meta = train_full_model(
        hp=hp,
        X_train_normal=X_train_normal,
        X_val_normal=X_val_normal,
        max_epochs=50,
        patience=10,
        device="cpu",
        seed=seed,
    )

    # 3. Calibrate Threshold on Normal validation errors
    val_normal_recon = model.reconstruct(X_val_normal)
    err_val_normal = np.mean((X_val_normal - val_normal_recon) ** 2, axis=1)

    threshold_p95 = float(np.percentile(err_val_normal, 95.0))
    threshold_p99 = float(np.percentile(err_val_normal, 99.0))
    selected_threshold = float(np.percentile(err_val_normal, threshold_percentile))

    val_normal_stats = {
        "n_normal_val": int(len(err_val_normal)),
        "error_mean": float(np.mean(err_val_normal)),
        "error_std": float(np.std(err_val_normal)),
        "error_p50": float(np.percentile(err_val_normal, 50)),
        "error_p95": threshold_p95,
        "error_p99": threshold_p99,
        "error_max": float(np.max(err_val_normal)),
    }

    # 4. Validation evaluation
    val_recon_all = model.reconstruct(split.X_val_scaled)
    val_scores = np.mean((split.X_val_scaled - val_recon_all) ** 2, axis=1)
    val_metrics = compute_anomaly_metrics(
        y_true_binary=y_val_binary,
        anomaly_scores=val_scores,
        threshold=selected_threshold,
    )

    # 5. Held-out Test evaluation
    lat_stats, thr_eps = measure_inference_performance(model, split.X_test_scaled)
    test_recon_all = model.reconstruct(split.X_test_scaled)
    test_scores = np.mean((split.X_test_scaled - test_recon_all) ** 2, axis=1)
    test_metrics = compute_anomaly_metrics(
        y_true_binary=y_test_binary,
        anomaly_scores=test_scores,
        threshold=selected_threshold,
        latency_ms=[lat_stats["mean"]],
        throughput_eps=thr_eps,
    )
    test_metrics["latency_ms"] = lat_stats

    total_experiment_duration = time.perf_counter() - start_exp_time

    # 6. Build Sample-level prediction dataframes
    val_preds_df = pd.DataFrame({
        "sample_index": np.arange(len(split.y_val)),
        "record_id": split.records_val,
        "true_label": split.y_val,
        "binary_label": y_val_binary,
        "reconstruction_error": val_scores,
        "predicted_binary": (val_scores > selected_threshold).astype(int),
    })

    test_preds_df = pd.DataFrame({
        "sample_index": np.arange(len(split.y_test)),
        "record_id": split.records_test,
        "true_label": split.y_test,
        "binary_label": y_test_binary,
        "reconstruction_error": test_scores,
        "predicted_binary": (test_scores > selected_threshold).astype(int),
    })

    # 7. Persist Artifacts
    run_id = f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    artifacts_dir = Path("artifacts/experiments") / experiment_name / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), artifacts_dir / "best_model.pt")

    # Save epoch history
    epoch_df.to_csv(artifacts_dir / "epoch_history.csv", index=False)

    # Save sample predictions
    val_preds_df.to_csv(artifacts_dir / "val_predictions.csv", index=False)
    test_preds_df.to_csv(artifacts_dir / "test_predictions.csv", index=False)

    # Save configuration
    config_payload = {
        "experiment_name": experiment_name,
        "optimizer_type": optimizer_type,
        "run_id": run_id,
        "hyperparameters": hp.to_dict(),
        "threshold_calibration": {
            "percentile": threshold_percentile,
            "threshold_value": selected_threshold,
            "threshold_p95": threshold_p95,
            "threshold_p99": threshold_p99,
            "normal_val_stats": val_normal_stats,
        },
        "split_strategy": split_strategy,
        "seed": seed,
    }
    (artifacts_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    # Save runtime metadata
    sys_meta = get_system_metadata()
    runtime_metadata = {
        **sys_meta,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "total_experiment_duration_s": total_experiment_duration,
        "optimization_duration_s": opt_time_s,
        "training_duration_s": train_meta["total_train_time_s"],
        "epochs_completed": train_meta["epochs_completed"],
        "best_epoch": train_meta["best_epoch"],
        "best_train_loss": train_meta["best_train_loss"],
        "best_val_loss": train_meta["best_val_loss"],
        "inference_latency_ms": lat_stats,
        "inference_throughput_eps": thr_eps,
    }
    (artifacts_dir / "runtime_metadata.json").write_text(json.dumps(runtime_metadata, indent=2))

    # Save search history if applicable
    if search_history is not None:
        (artifacts_dir / "search_history.json").write_text(json.dumps(search_history, indent=2))

    # Save comprehensive metrics JSON
    metrics_summary = {
        "experiment_name": experiment_name,
        "optimizer": optimizer_type.upper(),
        "run_id": run_id,
        "hyperparameters": hp.to_dict(),
        "threshold": selected_threshold,
        "threshold_percentile": threshold_percentile,
        "training_stats": train_meta,
        "optimization_stats": {
            "elapsed_s": opt_time_s,
            "evaluations": search_history["total_evaluations"] if search_history else 0,
            "best_fitness": search_history["best_fitness"] if search_history else val_metrics["auprc"],
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2))

    # Sync to latest symlink/directory
    latest_dir = Path("artifacts/experiments") / experiment_name / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["best_model.pt", "config.json", "epoch_history.csv", "metrics.json", "runtime_metadata.json"]:
        src = artifacts_dir / fname
        if src.exists():
            (latest_dir / fname).write_bytes(src.read_bytes())
    if search_history:
        (latest_dir / "search_history.json").write_bytes((artifacts_dir / "search_history.json").read_bytes())

    # Sync to legacy logs/training/experiments/<model_name>.json for dashboard backward compatibility
    dash_log_dir = Path("logs/training/experiments")
    dash_log_dir.mkdir(parents=True, exist_ok=True)
    dash_payload = {
        "experiment": experiment_name,
        "extra": {
            "optimizer": optimizer_type.upper(),
            "run_id": run_id,
            "hyperparameters": hp.to_dict(),
            "threshold": selected_threshold,
            "threshold_percentile": threshold_percentile,
            "validation_metrics": val_metrics,
            "optimization_time_s": opt_time_s,
            "total_train_time_s": train_meta["total_train_time_s"],
        },
        "metrics": {
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["f1_score"],
            "weighted_f1": test_metrics["f1_score"],
            "auroc": test_metrics["auroc"],
            "auprc": test_metrics["auprc"],
            "roc_auc_ovr_macro": test_metrics["auroc"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "specificity": test_metrics["specificity"],
            "mcc": test_metrics["mcc"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "latency_ms": lat_stats,
            "throughput_events_per_second": thr_eps,
        },
    }
    (dash_log_dir / f"{experiment_name}.json").write_text(json.dumps(dash_payload, indent=2))

    logger.info("COMPLETED: %s in %.2fs", experiment_name, total_experiment_duration)
    logger.info("Test Results: AUROC=%.4f | AUPRC=%.4f | F1=%.4f | Recall=%.4f | Specificity=%.4f | MCC=%.4f",
                test_metrics["auroc"], test_metrics["auprc"], test_metrics["f1_score"],
                test_metrics["recall"], test_metrics["specificity"], test_metrics["mcc"])

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Run Autoencoder Anomaly Detection experiments")
    parser.add_argument("--only", choices=["baseline_ae", "pso_ae", "gwo_ae", "coa_ae", "all"], default="all")
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    args = parser.parse_args()

    experiments = []
    if args.only in ["baseline_ae", "all"]:
        experiments.append(("baseline_ae", "none"))
    if args.only in ["pso_ae", "all"]:
        experiments.append(("pso_ae", "pso"))
    if args.only in ["gwo_ae", "all"]:
        experiments.append(("gwo_ae", "gwo"))
    if args.only in ["coa_ae", "all"]:
        experiments.append(("coa_ae", "coa"))

    all_results = {}
    for name, opt in experiments:
        res = execute_experiment(
            experiment_name=name,
            optimizer_type=opt,
            n_agents=args.n_agents,
            n_iterations=args.n_iterations,
            seed=args.seed,
            threshold_percentile=args.threshold_percentile,
        )
        all_results[name] = res

    # Write summary comparison JSON & CSV in reports/tables/
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "all_experiments_summary.json").write_text(json.dumps(all_results, indent=2))

    rows = []
    for name, r in all_results.items():
        tm = r["test_metrics"]
        vm = r["validation_metrics"]
        hp = r["hyperparameters"]
        rows.append({
            "Experiment": name,
            "Optimizer": r["optimizer"],
            "Hidden Dims": str(hp["hidden_dims"]),
            "Latent Dim": hp["latent_dim"],
            "Learning Rate": hp["lr"],
            "Weight Decay": hp["weight_decay"],
            "Dropout": hp["dropout"],
            "Noise Std": hp["noise_std"],
            "Batch Size": hp["batch_size"],
            "Threshold": r["threshold"],
            "Val AUROC": vm["auroc"],
            "Val AUPRC": vm["auprc"],
            "Val F1": vm["f1_score"],
            "Test AUROC": tm["auroc"],
            "Test AUPRC": tm["auprc"],
            "Test F1": tm["f1_score"],
            "Test Precision": tm["precision"],
            "Test Recall": tm["recall"],
            "Test Specificity": tm["specificity"],
            "Test Accuracy": tm["accuracy"],
            "Test MCC": tm["mcc"],
            "Test FPR": tm["false_positive_rate"],
            "FP / 1000 Normals": tm["fp_per_1000_normal"],
            "Latency Mean (ms)": tm["latency_ms"]["mean"],
            "Throughput (eps)": tm["throughput_events_per_second"],
            "Optimization Time (s)": r["optimization_stats"]["elapsed_s"],
            "Train Time (s)": r["training_stats"]["total_train_time_s"],
        })
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(tables_dir / "model_comparison_table.csv", index=False)
    logger.info("Saved consolidated comparison table to reports/tables/model_comparison_table.csv")
    print(comp_df.to_string())


if __name__ == "__main__":
    main()
