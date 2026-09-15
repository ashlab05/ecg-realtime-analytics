"""Single source of truth for ECG feature definitions and ordering.

The MIT-BIH Arrhythmia feature dataset contains 32 features per
heartbeat: 16 from Lead II (CSV prefix ``0_``) and 16 from Lead V
(CSV prefix ``1_``). Every component — training, Spark streaming,
FastAPI, and the dashboard — must agree on this exact ordering.

NOTE: the original project brief mentions "34 features / 17 per lead"
(including an ``rr`` field), but the actual dataset contains 32
features / 16 per lead: ``pre-RR`` and ``post-RR`` are the only RR
interval features present. This module reflects the real data.
"""

from __future__ import annotations

from typing import Dict, List

# Per-lead feature stems in canonical order (matches CSV column suffixes).
LEAD_FEATURE_STEMS: List[str] = [
    "pre-RR",
    "post-RR",
    "pPeak",
    "tPeak",
    "rPeak",
    "sPeak",
    "qPeak",
    "qrs_interval",
    "pq_interval",
    "qt_interval",
    "st_interval",
    "qrs_morph0",
    "qrs_morph1",
    "qrs_morph2",
    "qrs_morph3",
    "qrs_morph4",
]

# JSON payload (snake_case) key for each CSV stem.
STEM_TO_PAYLOAD_KEY: Dict[str, str] = {
    "pre-RR": "pre_rr",
    "post-RR": "post_rr",
    "pPeak": "p_peak",
    "tPeak": "t_peak",
    "rPeak": "r_peak",
    "sPeak": "s_peak",
    "qPeak": "q_peak",
    "qrs_interval": "qrs_interval",
    "pq_interval": "pq_interval",
    "qt_interval": "qt_interval",
    "st_interval": "st_interval",
    "qrs_morph0": "qrs_morph_0",
    "qrs_morph1": "qrs_morph_1",
    "qrs_morph2": "qrs_morph_2",
    "qrs_morph3": "qrs_morph_3",
    "qrs_morph4": "qrs_morph_4",
}

# Human-readable names for explanations and dashboard labels.
STEM_TO_DISPLAY_NAME: Dict[str, str] = {
    "pre-RR": "Pre-RR Interval",
    "post-RR": "Post-RR Interval",
    "pPeak": "P Peak",
    "tPeak": "T Peak",
    "rPeak": "R Peak",
    "sPeak": "S Peak",
    "qPeak": "Q Peak",
    "qrs_interval": "QRS Duration",
    "pq_interval": "PQ Interval",
    "qt_interval": "QT Interval",
    "st_interval": "ST Interval",
    "qrs_morph0": "QRS Morph 0",
    "qrs_morph1": "QRS Morph 1",
    "qrs_morph2": "QRS Morph 2",
    "qrs_morph3": "QRS Morph 3",
    "qrs_morph4": "QRS Morph 4",
}

LEAD_II_CSV_PREFIX = "0"   # Lead II columns in the CSV
LEAD_V_CSV_PREFIX = "1"    # Lead V columns in the CSV

LEAD_II_PAYLOAD_KEY = "lead_ii"
LEAD_V_PAYLOAD_KEY = "lead_v5"

# Canonical model feature order: all Lead II features, then all Lead V features.
FEATURE_COLUMNS: List[str] = (
    [f"{LEAD_II_CSV_PREFIX}_{stem}" for stem in LEAD_FEATURE_STEMS]
    + [f"{LEAD_V_CSV_PREFIX}_{stem}" for stem in LEAD_FEATURE_STEMS]
)

N_FEATURES: int = len(FEATURE_COLUMNS)  # 32

# Payload key -> canonical feature column, used to flatten JSON events.
PAYLOAD_TO_COLUMN: Dict[str, str] = {}
for _prefix_csv, _payload_lead in (
    (LEAD_II_CSV_PREFIX, LEAD_II_PAYLOAD_KEY),
    (LEAD_V_CSV_PREFIX, LEAD_V_PAYLOAD_KEY),
):
    for _stem in LEAD_FEATURE_STEMS:
        PAYLOAD_TO_COLUMN[f"{_payload_lead}.{STEM_TO_PAYLOAD_KEY[_stem]}"] = (
            f"{_prefix_csv}_{_stem}"
        )

# Canonical column -> display name, e.g. "0_qrs_interval" -> "QRS Duration (Lead II)".
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    col: (
        f"{STEM_TO_DISPLAY_NAME[col.split('_', 1)[1]]} "
        f"({'Lead II' if col.startswith(LEAD_II_CSV_PREFIX) else 'Lead V'})"
    )
    for col in FEATURE_COLUMNS
}


def flatten_payload(lead_ii: Dict[str, float], lead_v: Dict[str, float]) -> Dict[str, float]:
    """Flatten two per-lead payload dicts into {canonical_feature: value}.

    Raises KeyError if any expected feature key is missing.
    """
    flat: Dict[str, float] = {}
    for lead_payload, payload_lead_key in ((lead_ii, LEAD_II_PAYLOAD_KEY), (lead_v, LEAD_V_PAYLOAD_KEY)):
        for stem in LEAD_FEATURE_STEMS:
            key = STEM_TO_PAYLOAD_KEY[stem]
            if key not in lead_payload:
                raise KeyError(f"Missing payload key '{payload_lead_key}.{key}'")
            flat[PAYLOAD_TO_COLUMN[f"{payload_lead_key}.{key}"]] = float(lead_payload[key])
    if len(flat) != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {len(flat)}")
    return flat
