"""Health and readiness endpoints."""

from fastapi import APIRouter

from quant_signal.api.schemas import HealthResponse
from quant_signal.core.config import get_settings
from quant_signal.core.time import utc_now

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    """Return a liveness response."""

    settings = get_settings()
    return HealthResponse(status="ok", service=settings.app_name, timestamp=utc_now())


@router.get("/health/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    """Return a basic readiness response for the bootstrapped service."""

    settings = get_settings()
    return HealthResponse(status="ready", service=settings.app_name, timestamp=utc_now())
