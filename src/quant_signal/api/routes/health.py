"""Health and readiness endpoints."""

from fastapi import APIRouter, HTTPException, status

from quant_signal.api.schemas import HealthResponse
from quant_signal.core.config import get_settings
from quant_signal.core.time import utc_now
from quant_signal.storage.db import check_database_connection

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    """Return a liveness response."""

    settings = get_settings()
    return HealthResponse(status="ok", service=settings.app_name, timestamp=utc_now())


@router.get("/health/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    """Return a readiness response backed by the configured database."""

    settings = get_settings()
    try:
        check_database_connection(settings.database_url)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database connection unavailable",
        ) from exc
    return HealthResponse(status="ready", service=settings.app_name, timestamp=utc_now())
