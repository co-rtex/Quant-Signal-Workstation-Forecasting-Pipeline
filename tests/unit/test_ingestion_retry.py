"""Retry helper tests for ingestion."""

from __future__ import annotations

from datetime import date

import pytest

from quant_signal.core.config import Settings
from quant_signal.ingestion.errors import ProviderPermanentError, ProviderTransientError
from quant_signal.ingestion.models import MarketDataBar, ProviderFetchResult
from quant_signal.ingestion.retry import (
    compute_retry_delay,
    execute_provider_fetch_with_retry,
)


def test_compute_retry_delay_is_deterministic() -> None:
    """Backoff calculation should be exact and stable for known inputs."""

    settings = Settings(
        market_data_backoff_seconds=1.0,
        market_data_backoff_multiplier=2.0,
    )

    assert compute_retry_delay(1, settings) == 1.0
    assert compute_retry_delay(2, settings) == 2.0
    assert compute_retry_delay(3, settings) == 4.0


def test_compute_retry_delay_rejects_invalid_attempt_numbers() -> None:
    """Retry delay calculation should fail fast on invalid attempt numbers."""

    settings = Settings()

    with pytest.raises(ValueError, match="attempt_number must be >= 1"):
        compute_retry_delay(0, settings)


def test_execute_provider_fetch_with_retry_sleeps_then_succeeds() -> None:
    """Transient failures should schedule deterministic sleeps before succeeding."""

    settings = Settings(
        market_data_max_attempts=3,
        market_data_backoff_seconds=0.5,
        market_data_backoff_multiplier=3.0,
    )
    sleep_calls: list[float] = []
    bars = [
        MarketDataBar(
            symbol="AAPL",
            trade_date=date(2024, 1, 2),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            adjusted_close=100.5,
            volume=1_000_000,
        )
    ]
    attempt_counter = {"count": 0}

    def fetch_operation() -> ProviderFetchResult:
        attempt_counter["count"] += 1
        if attempt_counter["count"] == 1:
            raise ProviderTransientError(
                "static",
                "temporary upstream unavailable",
                cause_type="TimeoutError",
            )
        return ProviderFetchResult.from_bars(bars)

    result = execute_provider_fetch_with_retry(
        fetch_operation,
        provider_name="static",
        settings=settings,
        sleep_fn=sleep_calls.append,
    )

    assert result.terminal_error is None
    assert result.completed_after_retry is True
    assert result.fetch_result is not None
    assert sleep_calls == [0.5]
    assert result.attempt_log == [
        {
            "attempt_number": 1,
            "status": "failed",
            "retriable": True,
            "error_type": "TimeoutError",
            "error_message": "temporary upstream unavailable",
            "scheduled_backoff_seconds": 0.5,
        },
        {
            "attempt_number": 2,
            "status": "succeeded",
        },
    ]


def test_execute_provider_fetch_with_retry_does_not_sleep_for_permanent_failures() -> None:
    """Permanent failures should short-circuit even when multiple attempts are configured."""

    settings = Settings(
        market_data_max_attempts=3,
        market_data_backoff_seconds=1.0,
        market_data_backoff_multiplier=2.0,
    )
    sleep_calls: list[float] = []

    def fetch_operation() -> ProviderFetchResult:
        raise ProviderPermanentError(
            "static",
            "invalid provider response payload",
            cause_type="ValueError",
        )

    result = execute_provider_fetch_with_retry(
        fetch_operation,
        provider_name="static",
        settings=settings,
        sleep_fn=sleep_calls.append,
    )

    assert result.fetch_result is None
    assert result.completed_after_retry is False
    assert result.terminal_error is not None
    assert isinstance(result.terminal_error, ProviderPermanentError)
    assert sleep_calls == []
    assert result.attempt_log == [
        {
            "attempt_number": 1,
            "status": "failed",
            "retriable": False,
            "error_type": "ValueError",
            "error_message": "invalid provider response payload",
        }
    ]


def test_execute_provider_fetch_with_retry_skips_sleep_when_max_attempts_is_one() -> None:
    """Configured single-attempt fetches should not schedule backoff sleeps."""

    settings = Settings(
        market_data_max_attempts=1,
        market_data_backoff_seconds=1.0,
        market_data_backoff_multiplier=2.0,
    )
    sleep_calls: list[float] = []

    def fetch_operation() -> ProviderFetchResult:
        raise ProviderTransientError(
            "static",
            "temporary upstream unavailable",
            cause_type="TimeoutError",
        )

    result = execute_provider_fetch_with_retry(
        fetch_operation,
        provider_name="static",
        settings=settings,
        sleep_fn=sleep_calls.append,
    )

    assert result.fetch_result is None
    assert result.completed_after_retry is False
    assert result.terminal_error is not None
    assert isinstance(result.terminal_error, ProviderTransientError)
    assert sleep_calls == []
    assert result.attempt_log == [
        {
            "attempt_number": 1,
            "status": "failed",
            "retriable": True,
            "error_type": "TimeoutError",
            "error_message": "temporary upstream unavailable",
        }
    ]
