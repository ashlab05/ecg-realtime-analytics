#!/usr/bin/env python3
"""Train the tabular MLP 5-class baseline (Experiment 2 / ablation E).

Deep Learning comparison model only — not part of the production
inference pipeline.

Usage:
  python training/train_mlp.py [--epochs 60]
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
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_core.config import settings
from ecg_core.labels import CLASS_NAMES, LABEL_TO_INDEX
from ecg_core.metrics import compute_metrics, format_summary, save_experiment_results
from ecg_core.models.mlp import ECGMLPClassifier, save_mlp
from ecg_core.preprocessing import load_dataset, make_split
from ecg_core.training_utils import EarlyStopping, EpochLogger, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_mlp")


def train(epochs: int = 60, batch_size: int = 512, lr: float = 1e-3, strategy: str = "stratified", seed: int = 42) -> dict:
    seed_everything(seed)
    settings.ensure_dirs()

    df = load_dataset()
    split = make_split(df, strategy=strategy, seed=seed)

    to_idx = np.vectorize(LABEL_TO_INDEX.get)
    y_tr, y_va, y_te = to_idx(split.y_train), to_idx(split.y_val), to_idx(split.y_test)

    # Class-balanced sample weights for the imbalanced loss.
    w_tr = compute_sample_weight("balanced", y_tr)

    model = ECGMLPClassifier(n_features=split.X_train_scaled.shape[1], n_classes=len(CLASS_NAMES))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    train_ds = TensorDataset(
        torch.as_tensor(split.X_train_scaled, dtype=torch.float32),
        torch.as_tensor(y_tr, dtype=torch.long),
        torch.as_tensor(w_tr, dtype=torch.float32),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    epoch_logger = EpochLogger(model_name="mlp")
    stopper = EarlyStopping(patience=10)
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total, total_w = 0.0, 0.0
        for xb, yb, wb in train_dl:
            optimizer.zero_grad()
            per_sample = criterion(model(xb), yb)
            loss = (per_sample * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(xb)
            total_w += len(xb)
        train_loss = total / total_w

        model.eval()
        with torch.no_grad():
            va_logits = model(torch.as_tensor(split.X_val_scaled, dtype=torch.float32))
            va_loss = float(nn.functional.cross_entropy(va_logits, torch.as_tensor(y_va)).item())
            va_f1 = float(f1_score(y_va, va_logits.argmax(1).numpy(), average="macro"))

        epoch_logger.log(epoch, train_loss, va_loss, va_f1, lr, time.time() - start)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("Epoch %03d | train=%.5f val=%.5f val_macro_f1=%.4f", epoch, train_loss, va_loss, va_f1)
        if stopper.step(va_loss):
            logger.info("Early stopping at epoch %d", epoch)
            break

    # ---- Test evaluation ------------------------------------------------
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        probs = model.predict_proba(split.X_test_scaled)
    inference_time = time.time() - t0
    preds_idx = probs.argmax(1)
    preds = np.array([CLASS_NAMES[i] for i in preds_idx])
    latency_per_sample_ms = 1000.0 * inference_time / len(preds)

    metrics = compute_metrics(
        split.y_test, preds, probs=probs,
        latency_ms=[latency_per_sample_ms],
        throughput_eps=len(preds) / inference_time if inference_time > 0 else None,
    )
    logger.info(format_summary("mlp_baseline", metrics))

    save_mlp(model, {"val_macro_f1": va_f1, "test_macro_f1": metrics["macro_f1"]})
    save_experiment_results("mlp_baseline", metrics, extra={"split_strategy": strategy, "epochs_run": epoch})

    return {"macro_f1": metrics["macro_f1"], "model_dir": str(settings.mlp_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tabular MLP baseline")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--strategy", choices=["stratified", "record_aware"], default="stratified")
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    args = parser.parse_args()
    print(train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, strategy=args.strategy, seed=args.seed))


if __name__ == "__main__":
    main()
