"""Tabular MLP classifier baseline (PyTorch).

A Deep Learning 5-class baseline over the original 32 standardized
features. It is used for experimental comparison only — the production
pipeline uses Autoencoder -> XGBoost.
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
from ecg_core.labels import CLASS_NAMES

MODEL_WEIGHTS_FILE = "mlp.pt"
MODEL_META_FILE = "mlp_meta.json"


class ECGMLPClassifier(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_classes: int = len(CLASS_NAMES),
        hidden_dims: Optional[tuple] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers += [nn.Linear(dim, n_classes)]
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        logits = self(torch.as_tensor(np.asarray(X, dtype=np.float32)))
        return torch.softmax(logits, dim=1).numpy()


def save_mlp(model: ECGMLPClassifier, val_metrics: dict, out_dir: Optional[Path] = None) -> Path:
    out_dir = Path(out_dir or settings.mlp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / MODEL_WEIGHTS_FILE)
    meta = {
        "n_features": model.network[0].in_features,
        "n_classes": model.network[-1].out_features,
        "validation_metrics": val_metrics,
    }
    (out_dir / MODEL_META_FILE).write_text(json.dumps(meta, indent=2))
    return out_dir


def load_mlp(out_dir: Optional[Path] = None) -> ECGMLPClassifier:
    out_dir = Path(out_dir or settings.mlp_dir)
    weights = out_dir / MODEL_WEIGHTS_FILE
    meta_path = out_dir / MODEL_META_FILE
    if not weights.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Trained MLP not found in {out_dir}. Run `python training/train_mlp.py` first."
        )
    meta = json.loads(meta_path.read_text())
    model = ECGMLPClassifier(n_features=meta["n_features"], n_classes=meta["n_classes"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model
