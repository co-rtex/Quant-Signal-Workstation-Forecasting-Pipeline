"""Training artifact data structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    """Simple Platt-style probability calibrator."""

    strategy: str
    model: LogisticRegression | None

    @classmethod
    def fit(cls, probabilities: np.ndarray, y_true: np.ndarray) -> ProbabilityCalibrator:
        """Fit a calibrator from validation probabilities."""

        targets = y_true.astype(int)
        if np.unique(targets).size < 2 or targets.size < 20:
            return cls(strategy="identity", model=None)

        clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        model = LogisticRegression(random_state=42)
        model.fit(logits, targets)
        return cls(strategy="platt", model=model)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to raw positive-class probabilities."""

        clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
        if self.model is None:
            return np.asarray(clipped, dtype=float)

        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return np.asarray(self.model.predict_proba(logits)[:, 1], dtype=float)


@dataclass
class ModelArtifactBundle:
    """Serialized model artifact with preprocessing and calibration."""

    model_family: str
    feature_columns: list[str]
    target_column: str
    horizon_days: int
    dataset_version_id: str
    estimator: Any
    calibrator: ProbabilityCalibrator
    split_summary: dict[str, dict[str, date | int | str]]
    metadata_json: dict[str, object]

    def predict_positive_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict calibrated positive-class probabilities for a feature frame."""

        raw_probabilities = self.estimator.predict_proba(frame[self.feature_columns])[:, 1]
        return self.calibrator.predict(raw_probabilities)
