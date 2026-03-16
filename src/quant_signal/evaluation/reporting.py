"""Reporting helpers for model selection and evaluation summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def champion_sort_key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return a sort key implementing PR-AUC, Brier, then ROC-AUC selection."""

    pr_auc = metrics.get("pr_auc")
    brier_score = metrics.get("brier_score")
    roc_auc = metrics.get("roc_auc")

    return (
        -(float(pr_auc) if pr_auc is not None else float("-inf")),
        float(brier_score) if brier_score is not None else float("inf"),
        -(float(roc_auc) if roc_auc is not None else float("-inf")),
    )


def rank_candidate_metrics(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return candidate model records sorted by champion selection rules."""

    return sorted(records, key=lambda record: champion_sort_key(record["validation_metrics"]))
