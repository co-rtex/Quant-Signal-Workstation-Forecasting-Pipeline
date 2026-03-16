"""Market data ingestion orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from quant_signal.core.config import Settings, get_settings
from quant_signal.ingestion.providers import MarketDataProvider, YFinanceMarketDataProvider
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import IngestionRun
from quant_signal.storage.repositories import DailyBarRecord, StorageRepository


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
        self.provider = provider or YFinanceMarketDataProvider()

    def ingest_daily_bars(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> IngestionRun:
        """Fetch and persist normalized daily bars for the requested symbols."""

        requested = sorted({symbol.upper() for symbol in symbols})
        fetch_symbols = sorted({*requested, self.settings.benchmark_symbol.upper()})

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            run = repository.create_ingestion_run(
                provider=self.provider.name,
                start_date=start_date,
                end_date=end_date,
                requested_symbols=requested,
                metadata_json={"benchmark_symbol": self.settings.benchmark_symbol.upper()},
            )
            run_id = run.id

        try:
            bars = self.provider.fetch_daily_bars(fetch_symbols, start_date, end_date)
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
                    metadata_json={"fetched_symbols": fetch_symbols},
                )
        except Exception as exc:
            with session_scope(self.database_url) as session:
                repository = StorageRepository(session)
                repository.finalize_ingestion_run(
                    run_id=run_id,
                    status="failed",
                    records_written=0,
                    metadata_json={"error": str(exc)},
                )
            raise
