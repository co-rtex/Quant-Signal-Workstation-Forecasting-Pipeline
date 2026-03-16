"""FastAPI application factory."""

from fastapi import FastAPI

from quant_signal.api.routes.health import router as health_router
from quant_signal.api.routes.models import router as model_router
from quant_signal.api.routes.signals import router as signal_router
from quant_signal.core.config import get_settings
from quant_signal.core.logging import configure_logging


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Quant Signal Workstation",
        version="0.1.0",
        description="Production-minded quant forecasting platform.",
    )
    app.include_router(health_router)
    app.include_router(signal_router)
    app.include_router(model_router)
    return app


app = create_app()


def run() -> None:
    """Console entry point placeholder."""

    return None
