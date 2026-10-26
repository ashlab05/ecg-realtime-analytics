"""PyTorch autoencoder for normal-ECG representation learning (Stage 1).

Architecture (for the 32-feature dataset):

    32 -> 128 -> 64 -> 16 (latent) -> 64 -> 128 -> 32

The model is trained only on Normal (N) heartbeats; reconstruction
error on unseen beats is used as an anomaly score, with the decision
threshold calibrated as a percentile of validation-set normal errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from ecg_core.config import settings
from ecg_core.features import N_FEATURES


class ECGAutoencoder(nn.Module):
    """Fully-connected autoencoder over the standardized ECG feature vector."""

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_dims: Optional[tuple] = None,
        latent_dim: int = 16,
        dropout: float = 0.1,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.hidden_dims = tuple(hidden_dims) if hidden_dims else (128, 64)
        hidden_dims = self.hidden_dims
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.dropout = float(dropout)
        self.noise_std = float(noise_std)

        # Encoder: n_features -> ... -> latent_dim
        encoder_layers = []
        dim = n_features
        for h in hidden_dims:
            encoder_layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(self.dropout)]
            dim = h
        encoder_layers.append(nn.Linear(dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: latent_dim -> ... (reversed hidden dims) ... -> n_features
        decoder_layers = []
        dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(self.dropout)]
            dim = h
        decoder_layers.append(nn.Linear(dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(x) * self.noise_std
            return self.decoder(self.encoder(x + noise))
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    # ------------------------------------------------------------------
    # Inference helpers (numpy in / numpy out, for serving code)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return self(tensor).numpy()

    @torch.no_grad()
    def latent(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return self.encoder(tensor).numpy()

    @staticmethod
    def reconstruction_error(X: np.ndarray, X_recon: np.ndarray) -> np.ndarray:
        """Per-sample mean squared reconstruction error."""
        diff = np.asarray(X, dtype=np.float64) - np.asarray(X_recon, dtype=np.float64)
        return np.mean(diff**2, axis=1)


# ----------------------------------------------------------------------
# Persistence: weights (.pt) + metadata (architecture, threshold, metrics)
# ----------------------------------------------------------------------

MODEL_WEIGHTS_FILE = "autoencoder.pt"
MODEL_META_FILE = "autoencoder_meta.json"


def save_autoencoder(
    model: ECGAutoencoder,
    threshold: float,
    threshold_percentile: float,
    val_error_stats: dict,
    out_dir: Optional[Path] = None,
) -> Path:
    out_dir = Path(out_dir or settings.autoencoder_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / MODEL_WEIGHTS_FILE)
    meta = {
        "n_features": model.n_features,
        "hidden_dims": list(model.hidden_dims),
        "latent_dim": model.latent_dim,
        "dropout": getattr(model, "dropout", 0.1),
        "noise_std": getattr(model, "noise_std", 0.0),
        "threshold": float(threshold),
        "threshold_percentile": float(threshold_percentile),
        "validation_error_stats": val_error_stats,
    }
    (out_dir / MODEL_META_FILE).write_text(json.dumps(meta, indent=2))
    return out_dir


def load_autoencoder(out_dir: Optional[Path] = None, device: str = "cpu") -> ECGAutoencoder:
    """Load a trained autoencoder + threshold; raises FileNotFoundError if untrained."""
    out_dir = Path(out_dir or settings.autoencoder_dir)
    weights = out_dir / MODEL_WEIGHTS_FILE
    meta_path = out_dir / MODEL_META_FILE
    if not weights.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Trained autoencoder not found in {out_dir}. Run `python training/train_autoencoder.py` first."
        )
    meta = json.loads(meta_path.read_text())
    model = ECGAutoencoder(
        n_features=meta["n_features"],
        hidden_dims=tuple(meta.get("hidden_dims", (128, 64))),
        latent_dim=meta["latent_dim"],
        dropout=float(meta.get("dropout", 0.1)),
        noise_std=float(meta.get("noise_std", 0.0)),
    )
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_threshold(out_dir: Optional[Path] = None) -> float:
    out_dir = Path(out_dir or settings.autoencoder_dir)
    meta_path = out_dir / MODEL_META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Autoencoder metadata not found in {out_dir}; train the autoencoder first."
        )
    return float(json.loads(meta_path.read_text())["threshold"])
