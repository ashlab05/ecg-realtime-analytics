#!/usr/bin/env python3
"""Run the full experiment suite and produce a comparison report.

Experiments (see README section 'Research Evaluation'):
  A. xgboost_only      — 5-class XGBoost on the 32 original features.
  B. autoencoder_only  — anomaly detection (normal vs abnormal) via
                         reconstruction error threshold.
  C. hybrid_ae_xgb     — production pipeline: Stage 1 autoencoder
                         anomaly gate + Stage 2 XGBoost abnormal
                         classifier.
  D. xgboost_augmented — XGBoost on 32 features + 16 AE latent dims +
                         reconstruction error.
  E. mlp_baseline      — PyTorch tabular MLP, 5-class.

Each experiment is trained/evaluated on identical stratified splits
(seed=42) and reports accuracy, macro/weighted F1, per-class metrics,
confusion matrices, latency, and throughput. Results are saved to
logs/training/experiments/ with a final comparison_summary.json and
comparison_summary.csv.

Usage:
  python training/evaluate_models.py            # run everything
  python training/evaluate_models.py --only A C # run a subset
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xgboost as xgb

from ecg_core.config import settings
from ecg_core.labels import ABNORMAL_CLASSES, CLASS_NAMES, LABEL_TO_INDEX, NORMAL_CLASS
from ecg_core.metrics import compute_metrics, format_summary, save_experiment_results
from ecg_core.models.autoencoder import load_autoencoder, load_threshold
from ecg_core.models.mlp import load_mlp
from ecg_core.preprocessing import load_artifacts, load_dataset, make_split
from ecg_core.training_utils import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evaluate_models")

EXPERIMENTS = ["A", "B", "C", "D", "E"]
NAMES = {
    "A": "xgboost_only",
    "B": "autoencoder_only",
    "C": "hybrid_ae_xgb",
    "D": "xgboost_augmented",
    "E": "mlp_baseline",
}


def measure_latency_and_throughput(predict_fn, X: np.ndarray, repeats: int = 3) -> tuple:
    """Measure per-sample latency (ms) and throughput (events/s) on a feature matrix."""
    latencies, throughputs = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(X)
        dt = time.perf_counter() - t0
        latencies.append(1000.0 * dt / len(X))
        throughputs.append(len(X) / dt if dt > 0 else 0.0)
    return latencies, max(throughputs)


def run_xgboost_only(split, seed: int) -> dict:
    import training.train_xgboost as tx

    params = {**tx.DEFAULT_PARAMS, "random_state": seed}
    y_all_idx = np.array([LABEL_TO_INDEX[c] for c in split.y_train])
    unique, counts = np.unique(y_all_idx, return_counts=True)
    weight_map = {c: len(y_all_idx) / (len(unique) * cnt) for c, cnt in zip(unique, counts)}
    w_tr = np.array([weight_map[i] for i in y_all_idx])

    model = xgb.XGBClassifier(**params)
    model.fit(split.X_train_scaled, y_all_idx, sample_weight=w_tr, verbose=False)

    lat, thr = measure_latency_and_throughput(
        lambda X: model.predict(X), split.X_test_scaled
    )
    preds_idx = model.predict(split.X_test_scaled)
    preds = np.array([CLASS_NAMES[i] for i in preds_idx])
    metrics = compute_metrics(split.y_test, preds, probs=model.predict_proba(split.X_test_scaled),
                              class_names=CLASS_NAMES, latency_ms=lat, throughput_eps=thr)
    return metrics


def run_autoencoder_only(split) -> dict:
    ae = load_autoencoder()
    threshold = load_threshold()

    def predict(X: np.ndarray) -> np.ndarray:
        recon = ae.reconstruct(X)
        err = np.mean((X - recon) ** 2, axis=1)
        return np.where(err > threshold, "ABNORMAL", "N")

    lat, thr = measure_latency_and_throughput(predict, split.X_test_scaled)
    preds = predict(split.X_test_scaled)
    truth = np.where(split.y_test != NORMAL_CLASS, "ABNORMAL", "N")
    metrics = compute_metrics(truth, preds, class_names=["N", "ABNORMAL"], latency_ms=lat, throughput_eps=thr)
    return metrics


def run_hybrid(split, seed: int) -> dict:
    """Production pipeline: AE gate -> XGBoost Stage 2 for flagged samples."""
    ae = load_autoencoder()
    threshold = load_threshold()
    xgb_meta = json.loads((settings.xgboost_dir / "xgboost_meta.json").read_text())
    assert xgb_meta["mode"] == "abnormal", "Stage-2 model must be trained with --mode abnormal"
    stage2 = xgb.XGBClassifier()
    stage2.load_model(settings.xgboost_dir / "xgboost_model.json")
    stage2_classes = xgb_meta["classes"]

    def predict_one(X: np.ndarray):
        recon = ae.reconstruct(X)
        err = np.mean((X - recon) ** 2, axis=1)
        preds = np.full(len(X), NORMAL_CLASS, dtype=object)
        probs = np.zeros((len(X), len(CLASS_NAMES)))
        abnormal_mask = err > threshold
        if abnormal_mask.any():
            sub = X[abnormal_mask]
            idx2 = stage2.predict(sub)
            for row_pos, (i2, err_val) in enumerate(zip(idx2, err[abnormal_mask])):
                cls = stage2_classes[int(i2)]
                preds[np.where(abnormal_mask)[0][row_pos]] = cls
            p2 = stage2.predict_proba(sub)
            for j, cls in enumerate(stage2_classes):
                probs[abnormal_mask, LABEL_TO_INDEX[cls]] = p2[:, j]
        probs[~abnormal_mask, LABEL_TO_INDEX[NORMAL_CLASS]] = 1.0
        return preds, probs, err

    lat, thr = measure_latency_and_throughput(lambda X: predict_one(X)[0], split.X_test_scaled)
    preds, probs, _ = predict_one(split.X_test_scaled)
    metrics = compute_metrics(split.y_test, preds, probs=probs, class_names=CLASS_NAMES,
                              latency_ms=lat, throughput_eps=thr)
    # Fraction of samples routed to Stage 2 = computational overhead indicator.
    recon = ae.reconstruct(split.X_test_scaled)
    err = np.mean((split.X_test_scaled - recon) ** 2, axis=1)
    metrics["stage2_routing_rate"] = float((err > threshold).mean())
    return metrics


def run_xgboost_augmented(split, seed: int) -> dict:
    import training.train_xgboost as tx

    ae = load_autoencoder()

    def augment(X: np.ndarray) -> np.ndarray:
        latent, error = tx.build_augmented_features(X, ae)
        return np.hstack([X, latent, error])

    X_tr, X_te = augment(split.X_train_scaled), augment(split.X_test_scaled)
    y_tr_idx = np.array([LABEL_TO_INDEX[c] for c in split.y_train])
    unique, counts = np.unique(y_tr_idx, return_counts=True)
    weight_map = {c: len(y_tr_idx) / (len(unique) * cnt) for c, cnt in zip(unique, counts)}
    w_tr = np.array([weight_map[i] for i in y_tr_idx])

    params = {**tx.DEFAULT_PARAMS, "random_state": seed}
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr_idx, sample_weight=w_tr, verbose=False)

    lat, thr = measure_latency_and_throughput(lambda X: model.predict(X), X_te)
    preds = np.array([CLASS_NAMES[i] for i in model.predict(X_te)])
    metrics = compute_metrics(split.y_test, preds, probs=model.predict_proba(X_te),
                              class_names=CLASS_NAMES, latency_ms=lat, throughput_eps=thr)
    return metrics


def run_mlp_baseline(split) -> dict:
    mlp = load_mlp()

    def predict(X: np.ndarray) -> np.ndarray:
        return mlp.predict_proba(X).argmax(1)

    lat, thr = measure_latency_and_throughput(predict, split.X_test_scaled)
    preds_idx = predict(split.X_test_scaled)
    preds = np.array([CLASS_NAMES[i] for i in preds_idx])
    metrics = compute_metrics(split.y_test, preds, probs=mlp.predict_proba(split.X_test_scaled),
                              class_names=CLASS_NAMES, latency_ms=lat, throughput_eps=thr)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment suite A-E")
    parser.add_argument("--only", nargs="*", choices=EXPERIMENTS, default=EXPERIMENTS)
    parser.add_argument("--strategy", choices=["stratified", "record_aware"], default="stratified")
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    args = parser.parse_args()

    seed_everything(args.seed)
    settings.ensure_dirs()
    load_artifacts()  # fail early if preprocessing artifacts are missing

    df = load_dataset()
    split = make_split(df, strategy=args.strategy, seed=args.seed)
    logger.info("Loaded split (%s): train=%d val=%d test=%d", args.strategy, len(split.y_train), len(split.y_val), len(split.y_test))

    runners = {
        "A": ("xgboost_only", lambda: run_xgboost_only(split, args.seed)),
        "B": ("autoencoder_only", lambda: run_autoencoder_only(split)),
        "C": ("hybrid_ae_xgb", lambda: run_hybrid(split, args.seed)),
        "D": ("xgboost_augmented", lambda: run_xgboost_augmented(split, args.seed)),
        "E": ("mlp_baseline", lambda: run_mlp_baseline(split)),
    }

    all_results = {}
    for key in args.only:
        name, runner = runners[key]
        logger.info("=== Experiment %s: %s ===", key, name)
        metrics = runner()
        logger.info(format_summary(name, metrics))
        save_experiment_results(name, metrics, extra={"split_strategy": args.strategy, "seed": args.seed})
        all_results[name] = metrics

    # ---- Comparison summary ------------------------------------------------
    summary_rows = []
    for name, m in all_results.items():
        lat = m.get("latency_ms", {})
        summary_rows.append({
            "experiment": name,
            "accuracy": m.get("accuracy"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "macro_f1": m.get("macro_f1"),
            "weighted_f1": m.get("weighted_f1"),
            "sveb_recall": m.get("per_class", {}).get("SVEB", {}).get("recall"),
            "veb_recall": m.get("per_class", {}).get("VEB", {}).get("recall"),
            "f_recall": m.get("per_class", {}).get("F", {}).get("recall"),
            "q_recall": m.get("per_class", {}).get("Q", {}).get("recall"),
            "mean_latency_ms": lat.get("mean"),
            "throughput_eps": m.get("throughput_events_per_second"),
            "stage2_routing_rate": m.get("stage2_routing_rate"),
        })

    out_dir = settings.training_logs_dir / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary_rows, indent=2))
    with (out_dir / "comparison_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Comparison summary written to %s", out_dir / "comparison_summary.csv")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
