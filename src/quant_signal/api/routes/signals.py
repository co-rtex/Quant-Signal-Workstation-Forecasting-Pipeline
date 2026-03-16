"""Read-only signal serving endpoints."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status

from quant_signal.api.deps import get_signal_service
from quant_signal.api.schemas import RankedSignalResponse, SignalSnapshotResponse
from quant_signal.serving.service import SignalService

router = APIRouter(prefix="/v1", tags=["signals"])


@router.get("/signals", response_model=SignalSnapshotResponse)
def get_signals(
    as_of_date: date,
    horizon: int = Query(..., ge=1),
    limit: int = Query(default=10, ge=1, le=100),
    model_version_id: str | None = None,
    service: SignalService = Depends(get_signal_service),  # noqa: B008
) -> SignalSnapshotResponse:
    """Return ranked signals for a date and horizon."""

    signals = service.get_ranked_signals(as_of_date, horizon, limit, model_version_id)
    if not signals:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No signal snapshot found for the requested parameters.",
        )

    return SignalSnapshotResponse(
        as_of_date=signals[0].as_of_date,
        horizon_days=signals[0].horizon_days,
        model_version_id=signals[0].model_version_id,
        signals=[
            RankedSignalResponse(symbol=signal.symbol, score=signal.score, rank=signal.rank)
            for signal in signals
        ],
    )
