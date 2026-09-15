"""Shared training utilities: seeding, epoch CSV logging, early stopping."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ecg_core.config import settings


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpochLogger:
    """CSV epoch logger: epoch, train_loss, val_loss, val_f1, lr, elapsed_s."""

    COLUMNS = ["epoch", "train_loss", "val_loss", "val_f1", "lr", "elapsed_s"]

    def __init__(self, out_path: Optional[Path] = None, model_name: str = "model"):
        self.path = Path(out_path) if out_path else (
            settings.training_logs_dir / f"{model_name}_training_log.csv"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: List[dict] = []
        self._write_header()

    def _write_header(self) -> None:
        with self.path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

    def log(self, epoch: int, train_loss: float, val_loss: float, val_f1: float, lr: float, elapsed_s: float) -> None:
        row = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "val_f1": round(float(val_f1), 6),
            "lr": float(lr),
            "elapsed_s": round(float(elapsed_s), 3),
        }
        self._rows.append(row)
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)

    @property
    def rows(self) -> List[dict]:
        return list(self._rows)


class EarlyStopping:
    """Simple patience-based early stopping on validation loss."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
