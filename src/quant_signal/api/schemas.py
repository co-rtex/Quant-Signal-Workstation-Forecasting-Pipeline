"""API response schemas."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Standard health check payload."""

    status: str = Field(description="Current health status.")
    service: str = Field(description="Service name.")
    timestamp: datetime = Field(description="UTC timestamp for the health response.")


class RankedSignalResponse(BaseModel):
    """Individual ranked signal row."""

    symbol: str
    score: float
    rank: int


class SignalSnapshotResponse(BaseModel):
    """API payload for ranked signal snapshots."""

    as_of_date: date
    horizon_days: int
    model_version_id: str
    signals: list[RankedSignalResponse]


class ModelMetadataResponse(BaseModel):
    """API payload for model metadata."""

    model_version_id: str
    dataset_version_id: str
    horizon_days: int
    model_family: str
    champion_rank: int | None
    artifact_path: str
    feature_columns: list[str]
    metadata: dict[str, Any]


class EvaluationSplitResponse(BaseModel):
    """Evaluation summary for a single split."""

    split_name: str
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float | None
    calibration_error: float | None
    sample_count: int | None


class ModelEvaluationResponse(BaseModel):
    """API payload for model evaluation summaries."""

    model_version_id: str
    evaluations: list[EvaluationSplitResponse]
