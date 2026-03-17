"""Ingestion service and provider contract tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from quant_signal.core.config import Settings
from quant_signal.ingestion.errors import (
    ProviderPermanentError,
    ProviderTransientError,
    normalize_provider_error,
)
from quant_signal.ingestion.models import MarketDataBar, ProviderFetchResult
from quant_signal.ingestion.providers import (
    YFinanceMarketDataProvider,
    build_market_data_provider,
)
from quant_signal.ingestion.service import (
    build_retry_attempt_entry,
    build_retry_metadata,
    summarize_provider_fetch_result,
)


def test_build_market_data_provider_returns_yfinance() -> None:
    """Settings-backed provider factory should build the configured adapter."""

    provider = build_market_data_provider(Settings(market_data_provider="yfinance"))

    assert isinstance(provider, YFinanceMarketDataProvider)
    assert provider.name == "yfinance"


def test_build_market_data_provider_rejects_unknown_provider() -> None:
    """Unsupported provider names should fail fast."""

    settings = Settings(market_data_provider="unknown-provider")

    with pytest.raises(ValueError, match="Unsupported market data provider"):
        build_market_data_provider(settings)


def test_summarize_provider_fetch_result_tracks_missing_symbols_and_source_updates() -> None:
    """Fetch-result summaries should be JSON-safe and audit partial fetches."""

    bars = [
        MarketDataBar(
            symbol="AAPL",
            trade_date=date(2024, 1, 2),
            open=100.0,
            high=101.0,
            low=99.5,
            close=100.5,
            adjusted_close=100.5,
            volume=1_000_000,
            source_updated_at=datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
        ),
        MarketDataBar(
            symbol="AAPL",
            trade_date=date(2024, 1, 3),
            open=101.0,
            high=102.0,
            low=100.5,
            close=101.5,
            adjusted_close=101.5,
            volume=1_100_000,
            source_updated_at=datetime(2024, 1, 3, 21, 30, tzinfo=UTC),
        ),
    ]
    fetch_result = ProviderFetchResult.from_bars(
        bars,
        warnings=["partial fetch"],
    )

    summary = summarize_provider_fetch_result(fetch_result, ["AAPL", "SPY"])

    assert summary == {
        "returned_symbols": ["AAPL"],
        "missing_symbols": ["SPY"],
        "bar_count": 2,
        "per_symbol_bar_counts": {"AAPL": 2, "SPY": 0},
        "first_trade_date": "2024-01-02",
        "last_trade_date": "2024-01-03",
        "warnings": ["partial fetch"],
        "source_updated_at_min": "2024-01-02T21:00:00+00:00",
        "source_updated_at_max": "2024-01-03T21:30:00+00:00",
    }


def test_normalize_provider_error_classifies_transient_failures() -> None:
    """Timeout-like provider failures should be classified as transient."""

    error = normalize_provider_error("yfinance", TimeoutError("request timed out"))

    assert isinstance(error, ProviderTransientError)
    assert error.provider_name == "yfinance"
    assert error.cause_type == "TimeoutError"
    assert error.retriable is True


def test_normalize_provider_error_defaults_unknown_failures_to_permanent() -> None:
    """Unknown provider failures should not be marked retryable by default."""

    error = normalize_provider_error("yfinance", RuntimeError("unexpected failure"))

    assert isinstance(error, ProviderPermanentError)
    assert error.provider_name == "yfinance"
    assert error.cause_type == "RuntimeError"
    assert error.retriable is False


def test_build_retry_metadata_records_failed_attempt_shape() -> None:
    """Retry metadata should be JSON-safe and stable for failed fetch attempts."""

    settings = Settings(
        market_data_max_attempts=3,
        market_data_backoff_seconds=1.5,
        market_data_backoff_multiplier=2.0,
    )
    provider_error = ProviderTransientError(
        "yfinance",
        "request timed out",
        cause_type="TimeoutError",
    )
    attempt_log = [
        build_retry_attempt_entry(
            1,
            "failed",
            provider_error,
        )
    ]

    retry_metadata = build_retry_metadata(
        settings,
        attempt_log,
        completed_after_retry=False,
    )

    assert retry_metadata == {
        "configured_max_attempts": 3,
        "backoff_seconds": 1.5,
        "backoff_multiplier": 2.0,
        "attempt_count": 1,
        "completed_after_retry": False,
        "attempt_log": [
            {
                "attempt_number": 1,
                "status": "failed",
                "retriable": True,
                "error_type": "TimeoutError",
                "error_message": "request timed out",
            }
        ],
    }
