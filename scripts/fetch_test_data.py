#!/usr/bin/env python3
"""
Download and prepare the example fixture for SHAPExplainability.

Uses the UCI Parkinsons Voice Dataset (Little et al. 2007), downloaded
directly from the UCI Machine Learning Repository. The dataset contains
195 sustained phonations from 31 subjects (23 with Parkinson's disease,
8 healthy controls) with 22 biomedical voice measurements per recording
and a binary status label (1 = PD, 0 = healthy).

The 22 features are split into two conceptual modalities for the
plugin's modality-aggregation code:

    VC_  (vocal characteristics)
        16 features covering frequency (MDVP:Fo, Fhi, Flo), jitter
        (MDVP:Jitter*, RAP, PPQ, Jitter:DDP), shimmer (MDVP:Shimmer*,
        APQ3, APQ5, DDA), and harmonics-to-noise ratios (NHR, HNR).

    ND_  (nonlinear dynamics)
        6 features derived from nonlinear dynamical systems analysis:
        RPDE (recurrence period density entropy), DFA (detrended
        fluctuation analysis scaling exponent), spread1, spread2, D2
        (correlation dimension), PPE (pitch period entropy).

A RandomForestClassifier is trained with a fixed seed so the resulting
model is byte-identical across runs, giving deterministic SHAP outputs.

Reference:
    Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM (2007).
    Exploiting Nonlinear Recurrence and Fractal Scaling Properties for
    Voice Disorder Detection. Biomedical Engineering Online, 6:23.
    https://archive.ics.uci.edu/ml/datasets/parkinsons

Run from the repo root:

    python scripts/fetch_test_data.py

Outputs, under example/:
    features.csv    - (195, 22) feature matrix, indexed by recording id
    labels.csv      - (195, 1) binary status labels
    model.joblib    - trained RandomForestClassifier
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/parkinsons.data"
)

# Mapping from UCI column names to our prefixed, filesystem-safe names.
# Order matches UCI column order. The raw file has 24 columns: 'name',
# 22 feature columns, and 'status' (the label).
VC_FEATURES = [
    ("MDVP:Fo(Hz)", "VC_MDVP_Fo_Hz"),
    ("MDVP:Fhi(Hz)", "VC_MDVP_Fhi_Hz"),
    ("MDVP:Flo(Hz)", "VC_MDVP_Flo_Hz"),
    ("MDVP:Jitter(%)", "VC_MDVP_Jitter_pct"),
    ("MDVP:Jitter(Abs)", "VC_MDVP_Jitter_Abs"),
    ("MDVP:RAP", "VC_MDVP_RAP"),
    ("MDVP:PPQ", "VC_MDVP_PPQ"),
    ("Jitter:DDP", "VC_Jitter_DDP"),
    ("MDVP:Shimmer", "VC_MDVP_Shimmer"),
    ("MDVP:Shimmer(dB)", "VC_MDVP_Shimmer_dB"),
    ("Shimmer:APQ3", "VC_Shimmer_APQ3"),
    ("Shimmer:APQ5", "VC_Shimmer_APQ5"),
    ("MDVP:APQ", "VC_MDVP_APQ"),
    ("Shimmer:DDA", "VC_Shimmer_DDA"),
    ("NHR", "VC_NHR"),
    ("HNR", "VC_HNR"),
]

ND_FEATURES = [
    ("RPDE", "ND_RPDE"),
    ("DFA", "ND_DFA"),
    ("spread1", "ND_spread1"),
    ("spread2", "ND_spread2"),
    ("D2", "ND_D2"),
    ("PPE", "ND_PPE"),
]


def download_uci_parkinsons() -> pd.DataFrame:
    """Fetch the UCI Parkinsons data file and parse it into a DataFrame."""
    with urlopen(UCI_URL, timeout=30) as resp:
        raw_bytes = resp.read()
    return pd.read_csv(io.BytesIO(raw_bytes))


def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split UCI data into (features with modality prefixes, binary labels)."""
    vc_cols = [uci for uci, _ in VC_FEATURES]
    nd_cols = [uci for uci, _ in ND_FEATURES]
    missing = [c for c in vc_cols + nd_cols + ["name", "status"] if c not in raw.columns]
    if missing:
        raise RuntimeError(f"UCI file missing expected columns: {missing}")

    features_raw = raw[vc_cols + nd_cols].copy()
    rename_map = dict(VC_FEATURES + ND_FEATURES)
    features_raw.columns = [rename_map[c] for c in features_raw.columns]
    features_raw.index = raw["name"].values
    features_raw.index.name = ""

    labels = pd.DataFrame(
        {"label": raw["status"].astype(int).values},
        index=raw["name"].values,
    )
    labels.index.name = ""

    return features_raw, labels


def train_model(
    features: pd.DataFrame, labels: pd.DataFrame
) -> RandomForestClassifier:
    """Train a deterministic RandomForestClassifier."""
    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=5,
        random_state=42,
        n_jobs=1,
    )
    model.fit(features.values, labels["label"].values)
    return model


def main() -> None:
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading UCI Parkinsons dataset from {UCI_URL} ...")
    raw = download_uci_parkinsons()
    print(f"Got {raw.shape[0]} rows x {raw.shape[1]} columns.")

    features, labels = prepare_dataset(raw)
    model = train_model(features, labels)

    features_path = EXAMPLE_DIR / "features.csv"
    labels_path = EXAMPLE_DIR / "labels.csv"
    model_path = EXAMPLE_DIR / "model.joblib"

    features.to_csv(features_path)
    labels.to_csv(labels_path)
    joblib.dump(model, model_path)

    print(f"Wrote {features_path} ({features.shape})")
    print(f"Wrote {labels_path} ({labels.shape})")
    print(f"Wrote {model_path}")


if __name__ == "__main__":
    main()
