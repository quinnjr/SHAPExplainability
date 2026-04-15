#!/usr/bin/env python3
"""
Download and prepare the example fixture for SHAPExplainability.

Uses sklearn's breast_cancer dataset as a stand-in for a multi-omics
classification problem. The 30 real-valued features are split into two
synthetic modalities (MG_ and TX_) so the plugin's modality-aggregation
code has meaningful input. A RandomForestClassifier is trained with a
fixed random_state so the resulting model is byte-identical across runs.

Run from the repo root:

    python scripts/fetch_test_data.py

Outputs, under example/:
    features.csv    - (569, 30) feature matrix, indexed by sample id
    labels.csv      - (569, 1) binary labels, indexed by sample id
    model.joblib    - trained RandomForestClassifier
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example"


def prepare_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load breast_cancer and assign synthetic MG_/TX_ modality prefixes."""
    data = load_breast_cancer()
    raw_names = list(data.feature_names)

    half = len(raw_names) // 2
    renamed = [
        f"MG_{name.replace(' ', '_')}" if i < half else f"TX_{name.replace(' ', '_')}"
        for i, name in enumerate(raw_names)
    ]

    sample_ids = [f"Sample_{i:04d}" for i in range(data.data.shape[0])]
    features = pd.DataFrame(data.data, index=sample_ids, columns=renamed)
    labels = pd.DataFrame({"label": data.target}, index=sample_ids)
    return features, labels


def train_model(features: pd.DataFrame, labels: pd.DataFrame) -> RandomForestClassifier:
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

    features, labels = prepare_dataset()
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
