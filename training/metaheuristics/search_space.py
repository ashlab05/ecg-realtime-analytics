"""Unified hyperparameter search space for Autoencoder optimization.

Defines the continuous representation in [0, 1]^D and the decoding logic
into discrete/continuous model hyperparameters:
  0: latent_dim       -> choices [8, 12, 16, 24, 32]
  1: hidden_width     -> choices [64, 128, 192, 256] -> (width, max(32, width // 2))
  2: lr               -> log-scale in [1e-4, 1e-2]
  3: weight_decay     -> log-scale in [1e-6, 1e-2]
  4: dropout          -> continuous in [0.0, 0.30]
  5: noise_std        -> continuous in [0.0, 0.25] (denoising noise level)
  6: batch_size       -> choices [128, 256, 512]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


LATENT_CHOICES: List[int] = [8, 12, 16, 24, 32]
HIDDEN_WIDTH_CHOICES: List[int] = [64, 128, 192, 256]
BATCH_SIZE_CHOICES: List[int] = [128, 256, 512]
LR_MIN: float = 1e-4
LR_MAX: float = 1e-2
WD_MIN: float = 1e-6
WD_MAX: float = 1e-2
DROPOUT_MAX: float = 0.30
NOISE_STD_MAX: float = 0.25

DIMENSION: int = 7


@dataclass
class Hyperparameters:
    latent_dim: int
    hidden_dims: Tuple[int, ...]
    lr: float
    weight_decay: float
    dropout: float
    noise_std: float
    batch_size: int

    def to_dict(self) -> Dict:
        return {
            "latent_dim": int(self.latent_dim),
            "hidden_dims": list(self.hidden_dims),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "dropout": float(self.dropout),
            "noise_std": float(self.noise_std),
            "batch_size": int(self.batch_size),
        }


def decode_vector(vec: np.ndarray) -> Hyperparameters:
    """Decode a continuous normalized vector in [0, 1]^7 to Hyperparameters."""
    vec = np.clip(np.asarray(vec, dtype=float), 0.0, 1.0)

    # 0: latent_dim
    idx_latent = min(int(vec[0] * len(LATENT_CHOICES)), len(LATENT_CHOICES) - 1)
    latent_dim = LATENT_CHOICES[idx_latent]

    # 1: hidden_width
    idx_hw = min(int(vec[1] * len(HIDDEN_WIDTH_CHOICES)), len(HIDDEN_WIDTH_CHOICES) - 1)
    hw = HIDDEN_WIDTH_CHOICES[idx_hw]
    hidden_dims = (hw, max(32, hw // 2))

    # 2: lr (log-scale)
    log_lr = np.log10(LR_MIN) + vec[2] * (np.log10(LR_MAX) - np.log10(LR_MIN))
    lr = float(10 ** log_lr)

    # 3: weight_decay (log-scale)
    log_wd = np.log10(WD_MIN) + vec[3] * (np.log10(WD_MAX) - np.log10(WD_MIN))
    weight_decay = float(10 ** log_wd)

    # 4: dropout
    dropout = float(vec[4] * DROPOUT_MAX)

    # 5: noise_std (denoising level)
    noise_std = float(vec[5] * NOISE_STD_MAX)

    # 6: batch_size
    idx_bs = min(int(vec[6] * len(BATCH_SIZE_CHOICES)), len(BATCH_SIZE_CHOICES) - 1)
    batch_size = BATCH_SIZE_CHOICES[idx_bs]

    return Hyperparameters(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        noise_std=noise_std,
        batch_size=batch_size,
    )
