"""Candidate evaluator for Autoencoder Hyperparameter Optimization.

Trains an Autoencoder candidate exclusively on Normal training beats
and computes Validation AUPRC (Average Precision) on binary anomaly detection
as the fitness objective for the outer-loop metaheuristic.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_core.models.autoencoder import ECGAutoencoder
from ecg_core.training_utils import EarlyStopping
from training.metaheuristics.search_space import Hyperparameters

logger = logging.getLogger("evaluator")


def evaluate_candidate(
    hp: Hyperparameters,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    y_val_binary: np.ndarray,
    max_epochs: int = 15,
    patience: int = 3,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[float, Dict]:
    """Train a candidate autoencoder and return fitness (val AUPRC) plus info dict.

    Fitness = Validation AUPRC (Average Precision).
    Higher is better.
    """
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    stopper = EarlyStopping(patience=patience)
    best_loss = float("inf")
    best_weights = None
    best_epoch = 1

    val_tensor = torch.as_tensor(X_val, dtype=torch.float32).to(device)
    y_val_arr = np.asarray(y_val_binary, dtype=int)

    # Validation normal mask for reconstruction loss tracking
    val_normal_tensor = torch.as_tensor(X_val[y_val_arr == 0], dtype=torch.float32).to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_dl:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_recon = model(val_normal_tensor)
            val_loss = float(criterion(val_recon, val_normal_tensor).item())

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if stopper.step(val_loss):
            break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    # Compute reconstruction errors for all validation samples
    with torch.no_grad():
        all_recon = model(val_tensor).cpu().numpy()
    val_recon_errors = np.mean((X_val - all_recon) ** 2, axis=1)

    # Compute Validation AUPRC (fitness objective) and AUROC
    try:
        val_auprc = float(average_precision_score(y_val_arr, val_recon_errors))
    except Exception as e:
        logger.warning("Error computing AUPRC: %s", e)
        val_auprc = 0.0

    try:
        val_auroc = float(roc_auc_score(y_val_arr, val_recon_errors))
    except Exception:
        val_auroc = 0.0

    elapsed = time.perf_counter() - t0

    info = {
        "val_auprc": val_auprc,
        "val_auroc": val_auroc,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "eval_time_s": elapsed,
        "status": "success",
    }
    return val_auprc, info
