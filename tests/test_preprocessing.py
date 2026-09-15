"""Tests for preprocessing: schema validation, feature order, splits, artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg_core.features import FEATURE_COLUMNS, N_FEATURES
from ecg_core.labels import CLASS_NAMES
from ecg_core.preprocessing import (
    SchemaError,
    load_artifacts,
    make_split,
    save_artifacts,
    validate_dataframe,
)
from tests.conftest import make_synthetic_df


def test_feature_order_matches_dataset():
    """The canonical feature list must exactly match the real CSV columns."""
    df = pd.read_csv("data/raw/MIT-BIH Arrhythmia Database.csv", nrows=1)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    assert not missing
    assert N_FEATURES == 32
    assert df.shape[1] == N_FEATURES + 2  # + record + type


def test_validate_dataframe_ok(synthetic_df):
    out = validate_dataframe(synthetic_df)
    assert list(out["type"].unique()) == CLASS_NAMES or set(out["type"]) <= set(CLASS_NAMES)
    assert not out[FEATURE_COLUMNS].isna().any().any()


def test_validate_dataframe_missing_column(synthetic_df):
    bad = synthetic_df.drop(columns=[FEATURE_COLUMNS[0]])
    with pytest.raises(SchemaError, match="missing expected feature columns"):
        validate_dataframe(bad)


def test_validate_dataframe_drops_non_finite(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, FEATURE_COLUMNS[3]] = np.nan
    bad.loc[1, FEATURE_COLUMNS[4]] = np.inf
    out = validate_dataframe(bad)
    assert len(out) == len(bad) - 2


def test_validate_dataframe_unknown_label(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "type"] = "X"
    with pytest.raises(ValueError, match="Unknown ECG class label"):
        validate_dataframe(bad)


def test_stratified_split_no_leakage():
    df = make_synthetic_df(n_per_class=80, seed=3)
    split = make_split(df, strategy="stratified", seed=42)

    # Disjoint index sets by construction of X rows.
    n = len(df)
    assert split.X_train.shape[0] + split.X_val.shape[0] + split.X_test.shape[0] == n

    # Rows must be identical across splits (no duplicated rows).
    def rows(X):
        return {tuple(r) for r in np.round(X, 9)}

    train_rows = rows(split.X_train)
    assert len(train_rows & rows(split.X_val)) == 0
    assert len(train_rows & rows(split.X_test)) == 0

    # Every split must contain all 5 classes (stratification).
    for labels in (split.y_train, split.y_val, split.y_test):
        assert set(labels) == set(CLASS_NAMES)


def test_record_aware_split_disjoint_records():
    df = make_synthetic_df(n_per_class=80, seed=3)
    split = make_split(df, strategy="record_aware", seed=42)
    assert not (set(split.records_train) & set(split.records_test))
    assert not (set(split.records_train) & set(split.records_val))


def test_scaler_fit_on_train_only():
    """The scaler's mean must equal the training-split mean, not the full data."""
    df = make_synthetic_df(n_per_class=80, seed=5)
    split = make_split(df, strategy="stratified", seed=42)
    np.testing.assert_allclose(split.scaler.mean_, split.X_train.mean(axis=0), rtol=1e-9)


def test_artifacts_roundtrip(synthetic_df, tmp_path):
    split = make_split(synthetic_df, seed=11)
    from ecg_core.preprocessing import PreprocessingArtifacts

    artifacts = PreprocessingArtifacts(
        scaler=split.scaler,
        feature_columns=FEATURE_COLUMNS,
        feature_means=split.scaler.mean_,
        feature_stds=split.scaler.scale_,
    )
    save_artifacts(artifacts, out_dir=tmp_path / "preprocessing")

    # load_artifacts takes the *preprocessing dir itself* (not its parent).
    loaded = load_artifacts(tmp_path / "preprocessing")
    assert loaded.feature_columns == FEATURE_COLUMNS
    X = split.X_test[:5]
    np.testing.assert_allclose(loaded.transform(X), split.scaler.transform(X))


def test_artifacts_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_autoencoder"):
        load_artifacts(tmp_path)
