"""Tests for evaluation metrics."""

import numpy as np

from quant_signal.evaluation.metrics import compute_classification_metrics
from quant_signal.serving.service import rank_signal_frame


def test_compute_classification_metrics_returns_expected_fields() -> None:
    """Metric computation should return reliability and ranking fields."""

    metrics = compute_classification_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.3, 0.9]),
    )

    assert metrics["roc_auc"] is not None
    assert metrics["pr_auc"] is not None
    assert metrics["brier_score"] < 0.2
    assert metrics["sample_count"] == 4
    assert metrics["calibration_bins"]


def test_rank_signal_frame_orders_scores_within_each_date() -> None:
    """Signal ranking should sort descending by score within each date."""

    import pandas as pd

    ranked = rank_signal_frame(
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
                "symbol": ["MSFT", "AAPL", "AAPL"],
                "score": [0.8, 0.9, 0.7],
            }
        )
    )

    assert ranked.loc[0, "symbol"] == "AAPL"
    assert ranked.loc[0, "rank"] == 1
    assert ranked.loc[1, "rank"] == 2
