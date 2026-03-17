"""Market data ingestion orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

from quant_signal.core.config import Settings, get_settings
from quant_signal.ingestion.errors import ProviderError, normalize_provider_error
from quant_signal.ingestion.models import ProviderFetchResult
from quant_signal.ingestion.providers import (
    MarketDataProvider,
    build_market_data_provider,
)
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import IngestionRun
from quant_signal.storage.repositories import DailyBarRecord, StorageRepository


def build_provider_config_snapshot(settings: Settings) -> dict[str, object]:
    """Return the configured provider runtime knobs for persistence."""

    return {
        "max_attempts": int(settings.market_data_max_attempts),
        "backoff_seconds": float(settings.market_data_backoff_seconds),
        "backoff_multiplier": float(settings.market_data_backoff_multiplier),
    }


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
    attempt_log: Sequence[dict[str, object]],
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


def summarize_provider_fetch_result(
    fetch_result: ProviderFetchResult,
    fetch_symbols: Sequence[str],
) -> dict[str, object]:
    """Summarize provider fetch output into a JSON-serializable metadata shape."""

    normalized_fetch_symbols = sorted({symbol.upper() for symbol in fetch_symbols})
    bar_count = len(fetch_result.bars)

    returned_symbols = sorted(
        {
            *[symbol.upper() for symbol in fetch_result.returned_symbols],
            *[bar.symbol.upper() for bar in fetch_result.bars],
        }
    )
    per_symbol_bar_counts = {
        symbol: 0
        for symbol in normalized_fetch_symbols
    }
    for bar in fetch_result.bars:
        symbol = bar.symbol.upper()
        per_symbol_bar_counts[symbol] = per_symbol_bar_counts.get(symbol, 0) + 1

    trade_dates = [bar.trade_date for bar in fetch_result.bars]
    source_updated_at_values = [
        bar.source_updated_at
        for bar in fetch_result.bars
        if bar.source_updated_at is not None
    ]
    return {
        "returned_symbols": returned_symbols,
        "missing_symbols": [
            symbol
            for symbol in normalized_fetch_symbols
            if symbol not in returned_symbols
        ],
        "bar_count": bar_count,
        "per_symbol_bar_counts": per_symbol_bar_counts,
        "first_trade_date": _serialize_date(min(trade_dates) if trade_dates else None),
        "last_trade_date": _serialize_date(max(trade_dates) if trade_dates else None),
        "warnings": list(fetch_result.warnings),
        "source_updated_at_min": _serialize_datetime(
            min(source_updated_at_values) if source_updated_at_values else None
        ),
        "source_updated_at_max": _serialize_datetime(
            max(source_updated_at_values) if source_updated_at_values else None
        ),
    }


def _serialize_date(value: date | None) -> str | None:
    """Serialize an optional date for JSON persistence."""

    return value.isoformat() if value is not None else None


def _serialize_datetime(value: datetime | None) -> str | None:
    """Serialize an optional datetime for JSON persistence."""

    return value.isoformat() if value is not None else None


class IngestionService:
    """Orchestrate provider fetches and persisted raw bar writes."""

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
        settings: Settings | None = None,
        database_url: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database_url = database_url or self.settings.database_url
        self.provider = provider or build_market_data_provider(self.settings)

    def ingest_daily_bars(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> IngestionRun:
        """Fetch and persist normalized daily bars for the requested symbols."""

        requested = sorted({symbol.upper() for symbol in symbols})
        fetch_symbols = sorted({*requested, self.settings.benchmark_symbol.upper()})
        request_metadata = {
            "requested_symbols": requested,
            "fetch_symbols": fetch_symbols,
            "benchmark_symbol": self.settings.benchmark_symbol.upper(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        provider_metadata = {
            "name": self.provider.name,
            "config": build_provider_config_snapshot(self.settings),
        }
        initial_retry_metadata = build_retry_metadata(
            self.settings,
            [],
            completed_after_retry=False,
        )

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            run = repository.create_ingestion_run(
                provider=self.provider.name,
                start_date=start_date,
                end_date=end_date,
                requested_symbols=requested,
                metadata_json={
                    "request": request_metadata,
                    "provider": provider_metadata,
                    "retry": initial_retry_metadata,
                },
            )
            run_id = run.id

        try:
            fetch_result = self.provider.fetch_daily_bars(fetch_symbols, start_date, end_date)
            bars = fetch_result.bars
            records = [
                DailyBarRecord(
                    symbol=bar.symbol.upper(),
                    trade_date=bar.trade_date,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    adjusted_close=bar.adjusted_close,
                    volume=bar.volume,
                    source=self.provider.name,
                )
                for bar in bars
            ]

            with session_scope(self.database_url) as session:
                repository = StorageRepository(session)
                repository.upsert_symbols(
                    fetch_symbols,
                    benchmark_symbol=self.settings.benchmark_symbol.upper(),
                )
                row_count = repository.upsert_daily_bars(records, ingestion_run_id=run_id)
                return repository.finalize_ingestion_run(
                    run_id=run_id,
                    status="completed",
                    records_written=row_count,
                    metadata_json={
                        "provider": {
                            **provider_metadata,
                            "metadata": fetch_result.provider_metadata,
                        },
                        "retry": build_retry_metadata(
                            self.settings,
                            [build_retry_attempt_entry(1, "succeeded")],
                            completed_after_retry=False,
                        ),
                        "provider_fetch": summarize_provider_fetch_result(
                            fetch_result,
                            fetch_symbols,
                        ),
                        "persistence": {
                            "records_written": row_count,
                        },
                    },
                )
        except Exception as exc:
            with session_scope(self.database_url) as session:
                repository = StorageRepository(session)
                provider_error = normalize_provider_error(self.provider.name, exc)
                repository.finalize_ingestion_run(
                    run_id=run_id,
                    status="failed",
                    records_written=0,
                    metadata_json={
                        "provider": provider_metadata,
                        "retry": build_retry_metadata(
                            self.settings,
                            [
                                build_retry_attempt_entry(
                                    1,
                                    "failed",
                                    provider_error,
                                )
                            ],
                            completed_after_retry=False,
                        ),
                        "persistence": {"records_written": 0},
                        "failure": {
                            "error": str(provider_error),
                            "error_type": provider_error.cause_type,
                            "retriable": provider_error.retriable,
                        },
                    },
                )
            raise
