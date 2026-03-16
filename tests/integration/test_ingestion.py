"""Integration tests for ingestion and dataset materialization."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from quant_signal.core.config import Settings
from quant_signal.features.pipeline import FeaturePipeline
from quant_signal.ingestion.models import MarketDataBar
from quant_signal.ingestion.providers import MarketDataProvider
from quant_signal.ingestion.service import IngestionService
from quant_signal.storage.db import create_all_tables, session_scope
from quant_signal.storage.repositories import StorageRepository


class StaticProvider(MarketDataProvider):
    """Deterministic provider for integration tests."""

    name = "static"

    def __init__(self, bars: Sequence[MarketDataBar]) -> None:
        self._bars = list(bars)

    def fetch_daily_bars(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> list[MarketDataBar]:
        requested = {symbol.upper() for symbol in symbols}
        return [
            bar
            for bar in self._bars
            if bar.symbol.upper() in requested and start_date <= bar.trade_date <= end_date
        ]


def build_synthetic_bars() -> list[MarketDataBar]:
    """Build a deterministic synthetic market history."""

    bars: list[MarketDataBar] = []
    dates = pd.bdate_range("2024-01-02", periods=90)

    for index, timestamp in enumerate(dates):
        trade_date = timestamp.date()
        aapl_close = 100.0 + (index * 0.7) + ((index % 5) * 0.25)
        spy_close = 400.0 + (index * 0.3) + ((index % 7) * 0.15)

        bars.append(
            MarketDataBar(
                symbol="AAPL",
                trade_date=trade_date,
                open=aapl_close - 0.6,
                high=aapl_close + 0.8,
                low=aapl_close - 1.1,
                close=aapl_close,
                adjusted_close=aapl_close,
                volume=1_000_000 + (index * 1_000),
            )
        )
        bars.append(
            MarketDataBar(
                symbol="SPY",
                trade_date=trade_date,
                open=spy_close - 0.8,
                high=spy_close + 1.0,
                low=spy_close - 1.2,
                close=spy_close,
                adjusted_close=spy_close,
                volume=5_000_000 + (index * 2_000),
            )
        )

    return bars


def test_ingestion_and_dataset_pipeline(tmp_path: Path) -> None:
    """Ingestion should persist bars and feature pipeline should materialize a dataset artifact."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'integration.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
        default_horizons=[1, 5, 20],
    )
    create_all_tables(database_url)

    service = IngestionService(provider=StaticProvider(build_synthetic_bars()), settings=settings)
    run = service.ingest_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 5, 31))

    assert run.status == "completed"
    assert run.records_written > 0

    with session_scope(database_url) as session:
        repository = StorageRepository(session)
        bars = repository.load_daily_bars_frame(["AAPL", "SPY"])

    assert len(bars) == 180
    assert set(bars["symbol"].unique()) == {"AAPL", "SPY"}

    dataset = FeaturePipeline(settings=settings).build_dataset(date(2024, 5, 31), ["AAPL"])

    assert dataset.row_count > 0
    assert Path(dataset.artifact_path).exists()
    assert "momentum_5d" in dataset.feature_columns
    assert "target_up_20d" in dataset.label_columns
