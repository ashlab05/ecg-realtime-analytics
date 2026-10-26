"""Evaluation metrics and experiment logging shared by all training scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ecg_core.config import settings
from ecg_core.labels import CLASS_NAMES

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    latency_ms: Optional[List[float]] = None,
    throughput_eps: Optional[float] = None,
) -> Dict:
    """Compute the full metric suite for a classification experiment."""
    class_names = class_names or CLASS_NAMES
    present = [c for c in class_names if c in set(y_true) | set(y_pred)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
        for i, name in enumerate(class_names)
    }

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": confusion_matrix(y_true, y_pred, labels=class_names).tolist(),
        },
        "classes_present": present,
    }

    # ROC-AUC (one-vs-rest macro), only when probabilities are provided and
    # more than one class actually appears in y_true.
    if probs is not None:
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, probs, labels=class_names, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = None

    if latency_ms:
        arr = np.asarray(latency_ms, dtype=np.float64)
        metrics["latency_ms"] = {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    if throughput_eps is not None:
        metrics["throughput_events_per_second"] = float(throughput_eps)

    return metrics


def save_experiment_results(
    experiment_name: str,
    metrics: Dict,
    extra: Optional[Dict] = None,
) -> Path:
    """Persist experiment metrics as JSON under logs/training/experiments/."""
    out_dir = settings.training_logs_dir / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"experiment": experiment_name, **({"extra": extra} if extra else {}), "metrics": metrics}
    path = out_dir / f"{experiment_name}.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved experiment results to %s (macro_f1=%.4f)", path, metrics.get("macro_f1", float("nan")))
    return path


def format_summary(experiment_name: str, metrics: Dict) -> str:
    """One-line console summary of an experiment."""
    per_class = metrics.get("per_class", {})
    minority = " | ".join(
        f"{c}: R={per_class.get(c, {}).get('recall', 0):.3f}" for c in ("SVEB", "VEB", "F", "Q")
    )
    return (
        f"[{experiment_name}] acc={metrics.get('accuracy', 0):.4f} "
        f"macro_f1={metrics.get('macro_f1', 0):.4f} "
        f"weighted_f1={metrics.get('weighted_f1', 0):.4f} "
        f"balanced_acc={metrics.get('balanced_accuracy', 0):.4f} | {minority}"
    )


def compute_anomaly_metrics(
    y_true_binary: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    latency_ms: Optional[List[float]] = None,
    throughput_eps: Optional[float] = None,
) -> Dict:
    """Compute the full evaluation suite for binary ECG anomaly detection.

    Positive class (1) = Abnormal.
    Negative class (0) = Normal.
    Reconstruction error > threshold indicates Abnormal.
    Zero-division policy: return 0.0 for undefined metrics.
    """
    y_true = np.asarray(y_true_binary, dtype=int)
    scores = np.asarray(anomaly_scores, dtype=float)
    y_pred = (scores > threshold).astype(int)

    # Confusion matrix: TN, FP, FN, TP
    # labels=[0, 1] -> [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    n_normal = int(np.sum(y_true == 0))
    n_abnormal = int(np.sum(y_true == 1))
    total_samples = len(y_true)
    prevalence = float(n_abnormal / total_samples) if total_samples > 0 else 0.0

    # Sensitivity / Recall
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    # Specificity / True Negative Rate
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    # Precision
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    # F1-score
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    # Accuracy
    accuracy = float((tp + tn) / total_samples) if total_samples > 0 else 0.0
    # Balanced Accuracy
    balanced_acc = float((sensitivity + specificity) / 2.0)
    # False Positive Rate
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    # False positives per 1,000 normal samples
    fp_per_1000 = float(fpr * 1000.0)

    # Matthews correlation coefficient
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0

    # AUROC
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except Exception:
        auroc = 0.0

    # AUPRC / Average Precision
    try:
        auprc = float(average_precision_score(y_true, scores))
    except Exception:
        auprc = 0.0

    res: Dict = {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "labels": ["Normal (0)", "Abnormal (1)"],
            "matrix": cm.tolist(),
        },
        "false_positive_rate": fpr,
        "fp_per_1000_normal": fp_per_1000,
        "normal_samples": n_normal,
        "abnormal_samples": n_abnormal,
        "total_samples": total_samples,
        "class_prevalence": prevalence,
        "threshold_used": float(threshold),
        "positive_class": "abnormal (1)",
        "zero_division_policy": 0.0,
    }

    if latency_ms:
        arr = np.asarray(latency_ms, dtype=np.float64)
        res["latency_ms"] = {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    if throughput_eps is not None:
        res["throughput_events_per_second"] = float(throughput_eps)

    return res
