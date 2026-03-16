"""Time helpers."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(tz=UTC)
