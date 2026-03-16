"""API response schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Standard health check payload."""

    status: str = Field(description="Current health status.")
    service: str = Field(description="Service name.")
    timestamp: datetime = Field(description="UTC timestamp for the health response.")
