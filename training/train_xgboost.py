#!/usr/bin/env python3
"""Train XGBoost classifiers.

Supports three modes (used by the experiment pipeline):

  --mode baseline      5-class classifier on the original 32 features
                       (Experiment 1 / ablation A).
  --mode abnormal      4-class abnormal classifier (SVEB/VEB/F/Q) used as
                       Stage 2 of the production pipeline.
  --mode augmented     5-class classifier on 32 original features + 16
                       autoencoder latent features + reconstruction error
                       (Experiment 4 / ablation D).

Usage:
  python training/train_xgboost.py --mode abnormal
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import xgboost as xgb

from ecg_core.config import settings
from ecg_core.features import FEATURE_COLUMNS
from ecg_core.labels import (
    ABNORMAL_CLASSES,
    CLASS_NAMES,
    LABEL_TO_INDEX,
    NORMAL_CLASS,
)
from ecg_core.metrics import compute_metrics, format_summary, save_experiment_results
from ecg_core.models.autoencoder import load_autoencoder
from ecg_core.preprocessing import load_dataset, make_split, load_artifacts
from ecg_core.training_utils import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_xgboost")

MODEL_FILE = "xgboost_model.json"
META_FILE = "xgboost_meta.json"

# Class weights counteract the heavy imbalance (N=90k vs Q=15).
DEFAULT_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)


def build_augmented_features(X_scaled: np.ndarray, autoencoder) -> tuple:
    """Return (latent features, reconstruction error) for scaled input X."""
    latent = autoencoder.latent(X_scaled)
    recon = autoencoder.reconstruct(X_scaled)
    error = np.mean((X_scaled - recon) ** 2, axis=1, keepdims=True)
    return latent, error


def train(
    mode: str = "abnormal",
    strategy: str = "stratified",
    seed: int = 42,
) -> dict:
    assert mode in ("baseline", "abnormal", "augmented"), f"unknown mode {mode!r}"
    seed_everything(seed)
    settings.ensure_dirs()

    df = load_dataset()
    split = make_split(df, strategy=strategy, seed=seed)

    if mode == "abnormal":
        # Stage-2 model: train only on abnormal training samples.
        train_mask = split.y_train != NORMAL_CLASS
        val_mask = split.y_val != NORMAL_CLASS
        test_mask = split.y_test != NORMAL_CLASS
        X_tr, y_tr = split.X_train_scaled[train_mask], split.y_train[train_mask]
        X_va, y_va = split.X_val_scaled[val_mask], split.y_val[val_mask]
        X_te, y_te = split.X_test_scaled[test_mask], split.y_test[test_mask]
        classes = ABNORMAL_CLASSES
        out_dir = settings.xgboost_dir
    else:
        X_tr, y_tr = split.X_train_scaled, split.y_train
        X_va, y_va = split.X_val_scaled, split.y_val
        X_te, y_te = split.X_test_scaled, split.y_test
        classes = CLASS_NAMES
        # Baseline/augmented are experiment-only: keep them in subdirs so
        # they never clobber the production Stage-2 (abnormal) model at
        # models/xgboost/xgboost_model.json.
        out_dir = settings.xgboost_dir / mode

    if mode == "augmented":
        autoencoder = load_autoencoder()
        def augment(X: np.ndarray) -> np.ndarray:
            latent, error = build_augmented_features(X, autoencoder)
            return np.hstack([X, latent, error])

        X_tr, X_va, X_te = augment(X_tr), augment(X_va), augment(X_te)

    y_tr_idx = np.array([LABEL_TO_INDEX[c] for c in y_tr])
    y_va_idx = np.array([LABEL_TO_INDEX[c] for c in y_va])
    y_te_idx = np.array([LABEL_TO_INDEX[c] for c in y_te])

    # Map global class indices to a compact 0..k-1 range for XGBoost.
    class_list = classes if mode == "abnormal" else CLASS_NAMES
    remap = {LABEL_TO_INDEX[c]: i for i, c in enumerate(class_list)}
    y_tr_m = np.array([remap[i] for i in y_tr_idx])
    y_va_m = np.array([remap[i] for i in y_va_idx])
    y_te_m = np.array([remap[i] for i in y_te_idx])

    # Balanced sample weights (few Q samples exist).
    unique, counts = np.unique(y_tr_m, return_counts=True)
    weight_map = {c: len(y_tr_m) / (len(unique) * cnt) for c, cnt in zip(unique, counts)}
    w_tr = np.array([weight_map[c] for c in y_tr_m])

    logger.info(
        "Training XGBoost mode=%s | train=%d val=%d test=%d | classes=%s",
        mode, len(y_tr_m), len(y_va_m), len(y_te_m), class_list,
    )

    params = {**DEFAULT_PARAMS, "random_state": seed}
    model = xgb.XGBClassifier(**params)
    start = time.time()
    model.fit(
        X_tr,
        y_tr_m,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va_m)],
        verbose=False,
    )
    train_time = time.time() - start

    # ---- Test-set evaluation ------------------------------------------
    t0 = time.time()
    proba = model.predict_proba(X_te)
    preds_idx = model.predict(X_te)
    inference_time = time.time() - t0
    latency_per_sample_ms = 1000.0 * inference_time / max(len(X_te), 1)
    throughput = len(X_te) / inference_time if inference_time > 0 else None

    preds = np.array([class_list[i] for i in preds_idx])
    metrics = compute_metrics(
        y_te, preds, probs=proba, class_names=class_list,
        latency_ms=[latency_per_sample_ms], throughput_eps=throughput,
    )
    metrics["training_time_s"] = train_time

    logger.info(format_summary(f"xgboost_{mode}", metrics))

    # ---- Persist --------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / MODEL_FILE)
    meta = {
        "mode": mode,
        "classes": class_list,
        "n_features_in": int(X_tr.shape[1]),
        "feature_columns": FEATURE_COLUMNS,
        "augmented": mode == "augmented",
        "params": params,
        "test_metrics": metrics,
        "training_time_s": train_time,
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    if mode in ("baseline", "abnormal"):
        # Keep a joblib copy for convenience/tooling parity.
        joblib.dump(model, out_dir / "xgboost_model.joblib")
    logger.info("Saved XGBoost model to %s", out_dir)

    save_experiment_results(
        f"xgboost_{mode}",
        metrics,
        extra={"mode": mode, "split_strategy": strategy, "params": params},
    )
    return {"mode": mode, "macro_f1": metrics["macro_f1"], "model_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier")
    parser.add_argument(
        "--mode", choices=["baseline", "abnormal", "augmented"], default="abnormal",
        help="baseline=5-class on original features; abnormal=Stage-2 4-class; augmented=AE latent+error features",
    )
    parser.add_argument("--strategy", choices=["stratified", "record_aware"], default="stratified")
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    args = parser.parse_args()
    print(train(mode=args.mode, strategy=args.strategy, seed=args.seed))


if __name__ == "__main__":
    main()
