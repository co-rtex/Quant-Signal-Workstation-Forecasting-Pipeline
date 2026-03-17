"""Deterministic retry helpers for provider fetches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from quant_signal.core.config import Settings
from quant_signal.ingestion.errors import (
    ProviderError,
    ProviderTransientError,
    normalize_provider_error,
)
from quant_signal.ingestion.models import ProviderFetchResult

SleepFn = Callable[[float], None]
FetchOperation = Callable[[], ProviderFetchResult]


@dataclass(frozen=True)
class RetryExecutionResult:
    """Result of executing a provider fetch with deterministic retry behavior."""

    fetch_result: ProviderFetchResult | None
    attempt_log: list[dict[str, object]]
    completed_after_retry: bool
    terminal_error: ProviderError | None = None


def build_retry_attempt_entry(
    attempt_number: int,
    status: str,
    provider_error: ProviderError | None = None,
    *,
    scheduled_backoff_seconds: float | None = None,
) -> dict[str, object]:
    """Build a JSON-safe retry attempt entry."""

    entry: dict[str, object] = {
        "attempt_number": attempt_number,
        "status": status,
    }
    if provider_error is not None:
        entry["retriable"] = provider_error.retriable
        entry["error_type"] = provider_error.cause_type
        entry["error_message"] = str(provider_error)
    if scheduled_backoff_seconds is not None:
        entry["scheduled_backoff_seconds"] = float(scheduled_backoff_seconds)
    return entry


def build_retry_metadata(
    settings: Settings,
    attempt_log: list[dict[str, object]],
    *,
    completed_after_retry: bool,
) -> dict[str, object]:
    """Build the persisted retry metadata contract for an ingestion run."""

    return {
        "configured_max_attempts": int(settings.market_data_max_attempts),
        "backoff_seconds": float(settings.market_data_backoff_seconds),
        "backoff_multiplier": float(settings.market_data_backoff_multiplier),
        "attempt_count": len(attempt_log),
        "completed_after_retry": completed_after_retry,
        "attempt_log": list(attempt_log),
    }


def compute_retry_delay(attempt_number: int, settings: Settings) -> float:
    """Compute the deterministic delay before the retry after a failed attempt."""

    if attempt_number < 1:
        raise ValueError("attempt_number must be >= 1")

    base_delay = float(settings.market_data_backoff_seconds)
    multiplier = float(settings.market_data_backoff_multiplier)
    return base_delay * (multiplier ** (attempt_number - 1))


def execute_provider_fetch_with_retry(
    fetch_operation: FetchOperation,
    *,
    provider_name: str,
    settings: Settings,
    sleep_fn: SleepFn,
) -> RetryExecutionResult:
    """Execute a provider fetch with deterministic retries for transient failures."""

    attempt_log: list[dict[str, object]] = []
    max_attempts = int(settings.market_data_max_attempts)

    for attempt_number in range(1, max_attempts + 1):
        try:
            fetch_result = fetch_operation()
        except Exception as exc:
            provider_error = normalize_provider_error(provider_name, exc)
            should_retry = (
                isinstance(provider_error, ProviderTransientError)
                and attempt_number < max_attempts
            )
            scheduled_backoff_seconds = (
                compute_retry_delay(attempt_number, settings) if should_retry else None
            )
            attempt_log.append(
                build_retry_attempt_entry(
                    attempt_number,
                    "failed",
                    provider_error,
                    scheduled_backoff_seconds=scheduled_backoff_seconds,
                )
            )
            if should_retry:
                if scheduled_backoff_seconds is None:
                    raise RuntimeError(
                        "Retryable failure missing scheduled backoff"
                    ) from provider_error
                sleep_fn(scheduled_backoff_seconds)
                continue

            return RetryExecutionResult(
                fetch_result=None,
                attempt_log=attempt_log,
                completed_after_retry=False,
                terminal_error=provider_error,
            )

        attempt_log.append(build_retry_attempt_entry(attempt_number, "succeeded"))
        return RetryExecutionResult(
            fetch_result=fetch_result,
            attempt_log=attempt_log,
            completed_after_retry=attempt_number > 1,
        )

    raise RuntimeError("Retry execution exhausted without a terminal result")
