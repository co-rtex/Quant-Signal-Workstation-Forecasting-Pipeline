"""Model metadata endpoints."""

from fastapi import APIRouter, Depends

from quant_signal.api.deps import get_signal_service
from quant_signal.api.schemas import (
    EvaluationSplitResponse,
    ModelEvaluationResponse,
    ModelMetadataResponse,
)
from quant_signal.serving.service import SignalService

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("/{model_version}", response_model=ModelMetadataResponse)
def get_model_metadata(
    model_version: str,
    service: SignalService = Depends(get_signal_service),  # noqa: B008
) -> ModelMetadataResponse:
    """Return persisted model metadata."""

    model = service.get_model_version(model_version)
    return ModelMetadataResponse(
        model_version_id=model.id,
        dataset_version_id=model.dataset_version_id,
        horizon_days=model.horizon_days,
        model_family=model.model_family,
        champion_rank=model.champion_rank,
        artifact_path=model.artifact_path,
        feature_columns=model.feature_columns,
        metadata=model.metadata_json,
    )


@router.get("/{model_version}/evaluation", response_model=ModelEvaluationResponse)
def get_model_evaluation(
    model_version: str,
    service: SignalService = Depends(get_signal_service),  # noqa: B008
) -> ModelEvaluationResponse:
    """Return evaluation summaries for a model version."""

    evaluations = service.get_model_evaluations(model_version)
    def coerce_sample_count(evaluation_sample_count: object) -> int | None:
        if isinstance(evaluation_sample_count, (int, float)):
            return int(evaluation_sample_count)
        return None

    return ModelEvaluationResponse(
        model_version_id=model_version,
        evaluations=[
            EvaluationSplitResponse(
                split_name=evaluation.split_name,
                roc_auc=evaluation.roc_auc,
                pr_auc=evaluation.pr_auc,
                brier_score=evaluation.brier_score,
                calibration_error=evaluation.calibration_error,
                sample_count=coerce_sample_count(evaluation.metrics_json.get("sample_count")),
            )
            for evaluation in evaluations
        ],
    )
