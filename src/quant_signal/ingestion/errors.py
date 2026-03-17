"""Provider error taxonomy and normalization helpers."""

from __future__ import annotations

from typing import Final

TRANSIENT_MESSAGE_HINTS: Final[tuple[str, ...]] = (
    "timed out",
    "timeout",
    "temporary",
    "temporarily unavailable",
    "connection",
    "connection reset",
    "connection aborted",
    "rate limit",
    "too many requests",
    "429",
    "502",
    "503",
    "504",
)


class ProviderError(RuntimeError):
    """Base error raised by market data providers."""

    def __init__(
        self,
        provider_name: str,
        message: str,
        *,
        cause_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_name = provider_name
        self.cause_type = cause_type or self.__class__.__name__

    @property
    def retriable(self) -> bool:
        """Whether this provider failure is safe to retry."""

        return False


class ProviderTransientError(ProviderError):
    """Provider failure that may succeed on a later attempt."""

    @property
    def retriable(self) -> bool:
        """Transient provider failures are retryable."""

        return True


class ProviderPermanentError(ProviderError):
    """Provider failure that should not be retried."""


def normalize_provider_error(provider_name: str, exc: Exception) -> ProviderError:
    """Normalize arbitrary provider exceptions into the provider error taxonomy."""

    if isinstance(exc, ProviderError):
        return exc

    message = str(exc) or f"{provider_name} provider fetch failed"
    cause_type = exc.__class__.__name__
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return ProviderTransientError(provider_name, message, cause_type=cause_type)
    if isinstance(exc, (ValueError, KeyError, TypeError, AttributeError)):
        return ProviderPermanentError(provider_name, message, cause_type=cause_type)

    normalized_message = message.lower()
    if any(hint in normalized_message for hint in TRANSIENT_MESSAGE_HINTS):
        return ProviderTransientError(provider_name, message, cause_type=cause_type)

    return ProviderPermanentError(provider_name, message, cause_type=cause_type)
