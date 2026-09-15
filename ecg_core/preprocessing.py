"""Preprocessing pipeline: loading, validation, scaling, and splitting.

All artifacts (scaler, feature metadata) are persisted with joblib and
reused verbatim by the streaming and inference layers so training and
serving can never disagree on feature order or scaling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ecg_core.config import settings
from ecg_core.features import FEATURE_COLUMNS, N_FEATURES
from ecg_core.labels import CLASS_NAMES, LABEL_TO_INDEX, normalize_label

logger = logging.getLogger(__name__)

RECORD_COLUMN = "record"
LABEL_COLUMN = "type"


class SchemaError(ValueError):
    """Raised when the input data or event payload does not match the expected schema."""


@dataclass
class PreprocessingArtifacts:
    """Everything needed to transform a raw 32-dim vector into model input."""

    scaler: StandardScaler
    feature_columns: List[str]
    # Train-set per-feature statistics kept for reporting/diagnostics.
    feature_means: np.ndarray
    feature_stds: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.shape[-1] != N_FEATURES:
            raise SchemaError(
                f"Expected {N_FEATURES} features in order {FEATURE_COLUMNS[:2]}... got shape {X.shape}"
            )
        return self.scaler.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X)


# --------------------------------------------------------------------------
# Loading & validation
# --------------------------------------------------------------------------

def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the MIT-BIH feature CSV and validate its schema."""
    path = Path(path or settings.dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the MIT-BIH Arrhythmia feature "
            "CSV in data/raw/ or set ECG_DATASET_PATH."
        )
    df = pd.read_csv(path)
    return validate_dataframe(df)


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, labels, and numeric sanity of a feature dataframe."""
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SchemaError(f"Input data is missing expected feature columns: {missing_cols}")
    for col in (RECORD_COLUMN, LABEL_COLUMN):
        if col not in df.columns:
            raise SchemaError(f"Input data is missing required column '{col}'")

    df = df.copy()

    # Normalize labels (raises on unknown classes).
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(normalize_label)

    # Coerce features to numeric; non-numeric become NaN and are handled below.
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    n_missing = int(df[FEATURE_COLUMNS].isna().sum().sum())
    if n_missing:
        logger.warning("Found %d missing/non-numeric feature values; dropping affected rows.", n_missing)
        df = df.dropna(subset=FEATURE_COLUMNS)

    numeric = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    n_inf = int(np.isinf(numeric).sum())
    if n_inf:
        logger.warning("Found %d infinite feature values; dropping affected rows.", n_inf)
        keep = ~np.isinf(numeric).any(axis=1)
        df = df.loc[keep]

    # Constant / zero-variance features cannot be standardized — fail loudly.
    stds = df[FEATURE_COLUMNS].std()
    constant = stds[stds == 0].index.tolist()
    if constant:
        raise SchemaError(f"Constant feature columns detected (cannot standardize): {constant}")

    return df.reset_index(drop=True)


def frame_to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the (n_samples, 32) feature matrix in canonical column order."""
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

@dataclass
class SplitData:
    """A train/val/test split with labels and record ids kept alongside."""

    X_train: np.ndarray
    y_train: np.ndarray          # string labels
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    records_train: np.ndarray
    records_val: np.ndarray
    records_test: np.ndarray
    strategy: str                # "stratified" | "record_aware"
    scaler: StandardScaler

    @property
    def X_train_scaled(self) -> np.ndarray:
        return self.scaler.transform(self.X_train)

    @property
    def X_val_scaled(self) -> np.ndarray:
        return self.scaler.transform(self.X_val)

    @property
    def X_test_scaled(self) -> np.ndarray:
        return self.scaler.transform(self.X_test)


def make_split(
    df: pd.DataFrame,
    strategy: str = "stratified",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> SplitData:
    """Create a leakage-free split.

    strategy="stratified": random stratified split by class label.
    strategy="record_aware": hold out entire records (patients) so no
    patient appears in both train and test. Within train, val is drawn
    from training records only.

    The scaler is fit exclusively on the training split.
    """
    X = frame_to_matrix(df)
    y = df[LABEL_COLUMN].to_numpy()
    records = df[RECORD_COLUMN].to_numpy()

    if strategy == "stratified":
        idx = np.arange(len(df))
        idx_trainval, idx_test = train_test_split(
            idx, test_size=test_size, stratify=y, random_state=seed
        )
        # Validation is stratified on the remaining data.
        relative_val = val_size / (1.0 - test_size)
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=relative_val, stratify=y[idx_trainval], random_state=seed
        )
    elif strategy == "record_aware":
        rng = np.random.default_rng(seed)
        unique_records = np.unique(records)
        rng.shuffle(unique_records)
        n_records = len(unique_records)
        n_test = max(1, int(round(n_records * test_size)))
        n_val = max(1, int(round(n_records * val_size)))
        test_records = set(unique_records[:n_test])
        val_records = set(unique_records[n_test : n_test + n_val])
        is_test = np.isin(records, list(test_records))
        is_val = np.isin(records, list(val_records))
        is_train = ~is_test & ~is_val
        idx_train, idx_val, idx_test = (
            np.where(is_train)[0],
            np.where(is_val)[0],
            np.where(is_test)[0],
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy!r} (use 'stratified' or 'record_aware')")

    scaler = StandardScaler().fit(X[idx_train])

    return SplitData(
        X_train=X[idx_train],
        y_train=y[idx_train],
        X_val=X[idx_val],
        y_val=y[idx_val],
        X_test=X[idx_test],
        y_test=y[idx_test],
        records_train=records[idx_train],
        records_val=records[idx_val],
        records_test=records[idx_test],
        strategy=strategy,
        scaler=scaler,
    )


# --------------------------------------------------------------------------
# Artifact persistence
# --------------------------------------------------------------------------

def save_artifacts(artifacts: PreprocessingArtifacts, out_dir: Optional[Path] = None) -> Path:
    """Persist scaler + feature metadata to the preprocessing models dir."""
    out_dir = Path(out_dir or settings.preprocessing_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.scaler, out_dir / "scaler.joblib")
    meta = {
        "feature_columns": artifacts.feature_columns,
        "n_features": len(artifacts.feature_columns),
        "class_names": CLASS_NAMES,
        "label_to_index": LABEL_TO_INDEX,
    }
    (out_dir / "feature_metadata.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def load_artifacts(models_dir: Optional[Path] = None) -> PreprocessingArtifacts:
    """Load preprocessing artifacts; raises FileNotFoundError with guidance if missing."""
    pdir = Path(models_dir or settings.preprocessing_dir)
    scaler_path = pdir / "scaler.joblib"
    meta_path = pdir / "feature_metadata.json"
    if not scaler_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Preprocessing artifacts not found in {pdir}. "
            "Run `python training/train_autoencoder.py` (or the training pipeline) first."
        )
    scaler = joblib.load(scaler_path)
    meta = json.loads(meta_path.read_text())
    if meta["feature_columns"] != FEATURE_COLUMNS:
        raise SchemaError(
            "Saved feature metadata does not match the current feature definitions. "
            "Retrain the models to regenerate consistent artifacts."
        )
    return PreprocessingArtifacts(
        scaler=scaler,
        feature_columns=meta["feature_columns"],
        feature_means=scaler.mean_,
        feature_stds=scaler.scale_,
    )
