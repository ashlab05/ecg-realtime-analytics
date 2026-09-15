"""Shared test fixtures: synthetic dataset + tiny trained model artifacts.

Tests are fully self-contained — they train miniature models on
synthetic data in a temp directory, so the suite never depends on the
real dataset or on previously trained production artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# NOTE: xgboost MUST be imported before torch on macOS — importing torch
# first initializes a conflicting OpenMP runtime that segfaults inside
# xgboost's data dispatch (verified empirically on this stack).
import xgboost
import torch
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg_core.features import FEATURE_COLUMNS
from ecg_core.labels import CLASS_NAMES
from ecg_core.models.autoencoder import ECGAutoencoder, save_autoencoder
from ecg_core.preprocessing import PreprocessingArtifacts, save_artifacts
from training.train_xgboost import DEFAULT_PARAMS


def make_synthetic_df(n_per_class: int = 60, seed: int = 7) -> pd.DataFrame:
    """Synthetic feature dataframe: one Gaussian cluster per class."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, cls in enumerate(CLASS_NAMES):
        # Shift each class cluster along a few feature dimensions.
        mu = np.zeros(len(FEATURE_COLUMNS))
        mu[: 3 + i] = 0.8 * (i + 1)
        X = rng.normal(mu, 0.25, size=(n_per_class, len(FEATURE_COLUMNS)))
        df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        df.insert(0, "type", cls)
        df.insert(0, "record", f"{100 + i}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_df()


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory) -> Path:
    """Train miniature artifacts (autoencoder + scaler + XGBoost) in a temp dir."""
    root = tmp_path_factory.mktemp("models")

    df = make_synthetic_df()
    y = df["type"].to_numpy()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    # Split: 60/20/20 by class, scaler fit on train.
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    idx_train, idx_rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=7)
    idx_val, idx_test = train_test_split(idx_rest, test_size=0.5, stratify=y[idx_rest], random_state=7)
    scaler = StandardScaler().fit(X[idx_train])
    X_train_s = scaler.transform(X[idx_train])
    X_val_s = scaler.transform(X[idx_val])

    # --- Tiny autoencoder trained on normal training samples -------------
    X_n = X_train_s[y[idx_train] == "N"]
    ae = ECGAutoencoder(n_features=X.shape[1], hidden_dims=(16, 8), latent_dim=4)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    ds = TensorDataset(torch.as_tensor(X_n, dtype=torch.float32))
    for _ in range(15):
        for (batch,) in DataLoader(ds, batch_size=32, shuffle=True):
            opt.zero_grad()
            loss = nn.MSELoss()(ae(batch), batch)
            loss.backward()
            opt.step()
    ae.eval()
    with torch.no_grad():
        recon = ae(torch.as_tensor(X_val_s[y[idx_val] == "N"], dtype=torch.float32)).numpy()
    err = np.mean((X_val_s[y[idx_val] == "N"] - recon) ** 2, axis=1)
    threshold = float(np.percentile(err, 95))
    save_autoencoder(ae, threshold, 95.0, {"n_normal_val": len(err)}, out_dir=root / "autoencoder")

    save_artifacts(
        PreprocessingArtifacts(scaler=scaler, feature_columns=FEATURE_COLUMNS,
                               feature_means=scaler.mean_, feature_stds=scaler.scale_),
        out_dir=root / "preprocessing",
    )

    # --- Tiny Stage-2 XGBoost on abnormal training samples ----------------
    mask_tr = y[idx_train] != "N"
    X_ab, y_ab = X_train_s[mask_tr], y[idx_train][mask_tr]
    class_list = [c for c in CLASS_NAMES if c != "N"]
    remap = {c: i for i, c in enumerate(class_list)}
    clf = xgboost.XGBClassifier(**{**DEFAULT_PARAMS, "n_estimators": 20, "max_depth": 3})
    clf.fit(X_ab, np.array([remap[c] for c in y_ab]))
    xdir = root / "xgboost"
    xdir.mkdir(parents=True)
    clf.save_model(xdir / "xgboost_model.json")
    (xdir / "xgboost_meta.json").write_text(
        json.dumps({"mode": "abnormal", "classes": class_list,
                    "n_features_in": X.shape[1], "feature_columns": FEATURE_COLUMNS})
    )
    return root


@pytest.fixture(scope="session")
def test_bundle(models_dir):
    from inference.model_loader import load_bundle

    return load_bundle(models_dir)
