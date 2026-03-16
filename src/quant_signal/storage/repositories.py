"""Repository helpers for common persistence workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from quant_signal.storage.models import DailyBar, DatasetVersion, IngestionRun, Symbol


@dataclass(frozen=True)
class DailyBarRecord:
    """Normalized daily bar payload for persistence."""

    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    source: str


@dataclass(frozen=True)
class DatasetArtifactRecord:
    """Dataset artifact metadata to persist."""

    as_of_date: date
    feature_set_version: str
    horizons: list[int]
    symbols: list[str]
    row_count: int
    artifact_path: str
    artifact_hash: str
    feature_columns: list[str]
    label_columns: list[str]
    metadata_json: dict[str, object]


class StorageRepository:
    """Repository façade over common SQLAlchemy operations."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_symbols(
        self,
        symbols: Sequence[str],
        benchmark_symbol: str | None = None,
    ) -> dict[str, Symbol]:
        """Insert missing symbols and return a symbol map keyed by ticker."""

        normalized = sorted({symbol.upper() for symbol in symbols})
        if not normalized:
            return {}

        existing = self.session.execute(
            select(Symbol).where(Symbol.symbol.in_(normalized))
        ).scalars()
        symbol_map = {symbol.symbol: symbol for symbol in existing}

        for ticker in normalized:
            if ticker in symbol_map:
                symbol_map[ticker].is_benchmark = ticker == benchmark_symbol
                continue
            symbol = Symbol(symbol=ticker, is_benchmark=ticker == benchmark_symbol)
            self.session.add(symbol)
            symbol_map[ticker] = symbol

        self.session.flush()
        return symbol_map

    def create_ingestion_run(
        self,
        provider: str,
        start_date: date,
        end_date: date,
        requested_symbols: Sequence[str],
        metadata_json: dict[str, object] | None = None,
    ) -> IngestionRun:
        """Create a pending ingestion run."""

        run = IngestionRun(
            provider=provider,
            status="pending",
            start_date=start_date,
            end_date=end_date,
            requested_symbols=list(requested_symbols),
            metadata_json=metadata_json or {},
        )
        self.session.add(run)
        self.session.flush()
        return run

    def finalize_ingestion_run(
        self,
        run_id: str,
        status: str,
        records_written: int,
        metadata_json: dict[str, object] | None = None,
    ) -> IngestionRun:
        """Update the terminal state for an ingestion run."""

        run = self.session.get(IngestionRun, run_id)
        if run is None:
            raise ValueError(f"Unknown ingestion run: {run_id}")

        run.status = status
        run.records_written = records_written
        if metadata_json:
            run.metadata_json = {**run.metadata_json, **metadata_json}
        from quant_signal.core.time import utc_now

        run.completed_at = utc_now()
        self.session.flush()
        return run

    def upsert_daily_bars(
        self,
        records: Sequence[DailyBarRecord],
        ingestion_run_id: str | None = None,
    ) -> int:
        """Insert or update normalized daily bars."""

        if not records:
            return 0

        symbols = sorted({record.symbol.upper() for record in records})
        symbol_map = self.upsert_symbols(symbols)
        min_date = min(record.trade_date for record in records)
        max_date = max(record.trade_date for record in records)

        existing_rows = self.session.execute(
            select(DailyBar, Symbol.symbol)
            .join(Symbol, DailyBar.symbol_id == Symbol.id)
            .where(Symbol.symbol.in_(symbols))
            .where(DailyBar.trade_date >= min_date)
            .where(DailyBar.trade_date <= max_date)
        ).all()
        existing_map = {
            (symbol, row.trade_date, row.source): row
            for row, symbol in existing_rows
        }

        for record in records:
            key = (record.symbol.upper(), record.trade_date, record.source)
            current = existing_map.get(key)
            if current is None:
                current = DailyBar(
                    symbol_id=symbol_map[record.symbol.upper()].id,
                    ingestion_run_id=ingestion_run_id,
                    trade_date=record.trade_date,
                    open=record.open,
                    high=record.high,
                    low=record.low,
                    close=record.close,
                    adjusted_close=record.adjusted_close,
                    volume=record.volume,
                    source=record.source,
                )
                self.session.add(current)
                continue

            current.ingestion_run_id = ingestion_run_id
            current.open = record.open
            current.high = record.high
            current.low = record.low
            current.close = record.close
            current.adjusted_close = record.adjusted_close
            current.volume = record.volume

        self.session.flush()
        return len(records)

    def load_daily_bars_frame(
        self,
        symbols: Sequence[str],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Load normalized daily bars into a pandas DataFrame."""

        normalized = sorted({symbol.upper() for symbol in symbols})
        if not normalized:
            return pd.DataFrame()

        statement = (
            select(
                Symbol.symbol.label("symbol"),
                DailyBar.trade_date.label("date"),
                DailyBar.open,
                DailyBar.high,
                DailyBar.low,
                DailyBar.close,
                DailyBar.adjusted_close,
                DailyBar.volume,
                DailyBar.source,
            )
            .join(Symbol, DailyBar.symbol_id == Symbol.id)
            .where(Symbol.symbol.in_(normalized))
        )
        if start_date is not None:
            statement = statement.where(DailyBar.trade_date >= start_date)
        if end_date is not None:
            statement = statement.where(DailyBar.trade_date <= end_date)

        rows = self.session.execute(
            statement.order_by(DailyBar.trade_date, Symbol.symbol)
        ).mappings()
        return pd.DataFrame(rows)

    def create_dataset_version(self, record: DatasetArtifactRecord) -> DatasetVersion:
        """Persist a dataset artifact manifest."""

        dataset_version = DatasetVersion(
            as_of_date=record.as_of_date,
            feature_set_version=record.feature_set_version,
            horizons=record.horizons,
            symbols=record.symbols,
            row_count=record.row_count,
            artifact_path=record.artifact_path,
            artifact_hash=record.artifact_hash,
            feature_columns=record.feature_columns,
            label_columns=record.label_columns,
            metadata_json=record.metadata_json,
        )
        self.session.add(dataset_version)
        self.session.flush()
        return dataset_version

    def get_dataset_version(self, dataset_version_id: str) -> DatasetVersion:
        """Return a dataset version by id."""

        dataset_version = self.session.get(DatasetVersion, dataset_version_id)
        if dataset_version is None:
            raise ValueError(f"Unknown dataset version: {dataset_version_id}")
        return dataset_version
