"""Class label definitions and encoding for MIT-BIH arrhythmia classes."""

from __future__ import annotations

from typing import Dict, List

# Canonical class order used across all models and reports.
CLASS_NAMES: List[str] = ["N", "SVEB", "VEB", "F", "Q"]

CLASS_DESCRIPTIONS: Dict[str, str] = {
    "N": "Normal beat",
    "SVEB": "Supraventricular ectopic beat",
    "VEB": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unknown beat",
}

# Label -> integer index (consistent across autoencoder stage, XGBoost, MLP).
LABEL_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}

NORMAL_CLASS = "N"
# Abnormal classes handled by the Stage-2 XGBoost classifier.
ABNORMAL_CLASSES: List[str] = ["SVEB", "VEB", "F", "Q"]

# Some dataset variants use full names; map anything we might encounter.
LABEL_ALIASES: Dict[str, str] = {
    "N": "N",
    "NORMAL": "N",
    "S": "SVEB",
    "SVEB": "SVEB",
    "V": "VEB",
    "VEB": "VEB",
    "F": "F",
    "FUSION": "F",
    "Q": "Q",
    "UNKNOWN": "Q",
}


def normalize_label(raw: str) -> str:
    """Normalize a raw label to one of CLASS_NAMES; raises ValueError otherwise."""
    key = str(raw).strip().upper()
    if key not in LABEL_ALIASES:
        raise ValueError(f"Unknown ECG class label: {raw!r} (expected one of {CLASS_NAMES})")
    return LABEL_ALIASES[key]


def is_abnormal(label: str) -> bool:
    return normalize_label(label) != NORMAL_CLASS
