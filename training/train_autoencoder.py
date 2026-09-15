#!/usr/bin/env python3
"""Train the Stage-1 autoencoder on Normal (N) ECG heartbeats.

Steps:
  1. Load and validate the MIT-BIH feature dataset.
  2. Stratified split (train/val/test) — scaler fit on train only.
  3. Train autoencoder ONLY on Normal samples from the train split.
  4. Calibrate the anomaly threshold as the 95th percentile of
     reconstruction errors on Normal validation samples.
  5. Save model, threshold, scaler, and training logs.

Usage:
  python training/train_autoencoder.py [--epochs 50] [--strategy stratified]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_core.config import settings
from ecg_core.labels import NORMAL_CLASS
from ecg_core.models.autoencoder import ECGAutoencoder, save_autoencoder
from ecg_core.preprocessing import load_dataset, make_split, save_artifacts, PreprocessingArtifacts
from ecg_core.features import FEATURE_COLUMNS
from ecg_core.training_utils import EarlyStopping, EpochLogger, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_autoencoder")


def train(
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    threshold_percentile: float = 95.0,
    strategy: str = "stratified",
    seed: int = 42,
) -> dict:
    seed_everything(seed)
    settings.ensure_dirs()

    df = load_dataset()
    logger.info("Loaded dataset: %s", df.shape)
    split = make_split(df, strategy=strategy, seed=seed)
    logger.info("Split (%s): train=%d val=%d test=%d", strategy, len(split.y_train), len(split.y_val), len(split.y_test))

    # ---- Autoencoder trains on NORMAL training samples only -------------
    X_train_n = split.X_train_scaled[split.y_train == NORMAL_CLASS]
    X_val_n = split.X_val_scaled[split.y_val == NORMAL_CLASS]
    logger.info("Normal training samples: %d | normal validation samples: %d", len(X_train_n), len(X_val_n))

    model = ECGAutoencoder(n_features=X_train_n.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.as_tensor(X_train_n, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    epoch_logger = EpochLogger(model_name="autoencoder")
    stopper = EarlyStopping(patience=10)
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_recon = model(torch.as_tensor(X_val_n, dtype=torch.float32)).numpy()
        val_loss = float(np.mean((X_val_n - val_recon) ** 2))

        # Validation F1 on the anomaly-detection task (normal vs abnormal)
        # computed with the *current* validation normal error distribution.
        err_val_n = np.mean((X_val_n - val_recon) ** 2, axis=1)
        thr = float(np.percentile(err_val_n, threshold_percentile))
        all_val = split.X_val_scaled
        with torch.no_grad():
            recon_all = model(torch.as_tensor(all_val, dtype=torch.float32)).numpy()
        err_all = np.mean((all_val - recon_all) ** 2, axis=1)
        pred_abnormal = err_all > thr
        true_abnormal = split.y_val != NORMAL_CLASS
        val_f1 = float(f1_score(true_abnormal, pred_abnormal))

        epoch_logger.log(epoch, train_loss, val_loss, val_f1, lr, time.time() - start)
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %03d | train=%.6f val=%.6f val_anomaly_f1=%.4f", epoch, train_loss, val_loss, val_f1
            )
        if stopper.step(val_loss):
            logger.info("Early stopping at epoch %d (best val=%.6f)", epoch, stopper.best)
            break

    # ---- Threshold calibration on normal validation samples -------------
    model.eval()
    with torch.no_grad():
        val_recon = model(torch.as_tensor(X_val_n, dtype=torch.float32)).numpy()
    err_val_n = ECGAutoencoder.reconstruction_error(X_val_n, val_recon)
    threshold = float(np.percentile(err_val_n, threshold_percentile))

    val_stats = {
        "n_normal_val": int(len(err_val_n)),
        "error_mean": float(err_val_n.mean()),
        "error_std": float(err_val_n.std()),
        "error_p50": float(np.percentile(err_val_n, 50)),
        "error_p95": float(np.percentile(err_val_n, 95)),
        "error_p99": float(np.percentile(err_val_n, 99)),
        "error_max": float(err_val_n.max()),
    }
    logger.info("Threshold (p%g of normal val errors): %.6f", threshold_percentile, threshold)

    # ---- Persist everything ----------------------------------------------
    save_autoencoder(model, threshold, threshold_percentile, val_stats)
    artifacts = PreprocessingArtifacts(
        scaler=split.scaler,
        feature_columns=FEATURE_COLUMNS,
        feature_means=split.scaler.mean_,
        feature_stds=split.scaler.scale_,
    )
    save_artifacts(artifacts)
    logger.info("Saved autoencoder + preprocessing artifacts to %s", settings.autoencoder_dir)

    return {"threshold": threshold, "val_error_stats": val_stats, "epochs_run": epoch_logger.rows[-1]["epoch"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ECG normal-pattern autoencoder")
    parser.add_argument("--epochs", type=int, default=settings.ae_epochs)
    parser.add_argument("--batch-size", type=int, default=settings.ae_batch_size)
    parser.add_argument("--lr", type=float, default=settings.ae_learning_rate)
    parser.add_argument("--threshold-percentile", type=float, default=settings.ae_threshold_percentile)
    parser.add_argument("--strategy", choices=["stratified", "record_aware"], default="stratified")
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    args = parser.parse_args()

    result = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threshold_percentile=args.threshold_percentile,
        strategy=args.strategy,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
