"""Evaluation metrics for probabilistic classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def compute_calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> tuple[list[dict[str, float]], float]:
    """Return calibration bins and expected calibration error."""

    frame = pd.DataFrame(
        {
            "y_true": y_true.astype(float),
            "probability": np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6),
        }
    )
    frame["bucket"] = pd.cut(
        frame["probability"],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
    )

    bins: list[dict[str, float]] = []
    weighted_error = 0.0

    for index, interval in enumerate(frame["bucket"].cat.categories):
        bucket_rows = frame[frame["bucket"] == interval]
        if bucket_rows.empty:
            continue

        mean_prediction = float(bucket_rows["probability"].mean())
        observed_rate = float(bucket_rows["y_true"].mean())
        count = float(len(bucket_rows))
        weighted_error += abs(mean_prediction - observed_rate) * count
        bins.append(
            {
                "bin": float(index),
                "lower": float(interval.left),
                "upper": float(interval.right),
                "count": count,
                "mean_prediction": mean_prediction,
                "observed_rate": observed_rate,
            }
        )

    calibration_error = weighted_error / max(float(len(frame)), 1.0)
    return bins, calibration_error


def compute_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute reliability-focused classification metrics."""

    y_true_int = y_true.astype(int)
    clipped_probabilities = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    unique_classes = np.unique(y_true_int)

    roc_auc: float | None = None
    pr_auc: float | None = None
    if unique_classes.size > 1:
        roc_auc = float(roc_auc_score(y_true_int, clipped_probabilities))
        pr_auc = float(average_precision_score(y_true_int, clipped_probabilities))

    calibration_bins, calibration_error = compute_calibration_bins(
        y_true_int,
        clipped_probabilities,
    )
    brier_score = float(brier_score_loss(y_true_int, clipped_probabilities))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier_score,
        "calibration_error": calibration_error,
        "positive_rate": float(y_true_int.mean()),
        "sample_count": int(y_true_int.size),
        "calibration_bins": calibration_bins,
    }
