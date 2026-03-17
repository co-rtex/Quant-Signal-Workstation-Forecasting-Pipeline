"""Integration tests for ingestion and dataset materialization."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from quant_signal.core.config import Settings
from quant_signal.features.pipeline import FeaturePipeline
from quant_signal.ingestion.errors import ProviderPermanentError, ProviderTransientError
from quant_signal.ingestion.models import MarketDataBar, ProviderFetchResult
from quant_signal.ingestion.providers import MarketDataProvider
from quant_signal.ingestion.service import IngestionService
from quant_signal.storage.db import create_all_tables, session_scope
from quant_signal.storage.models import IngestionRun
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
    ) -> ProviderFetchResult:
        requested = {symbol.upper() for symbol in symbols}
        filtered_bars = [
            bar
            for bar in self._bars
            if bar.symbol.upper() in requested and start_date <= bar.trade_date <= end_date
        ]
        return ProviderFetchResult.from_bars(
            filtered_bars,
            provider_metadata={"fixture": "synthetic"},
        )


class TransientFailingProvider(MarketDataProvider):
    """Provider that fails with a retryable provider error."""

    name = "transient-static"

    def fetch_daily_bars(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> ProviderFetchResult:
        raise ProviderTransientError(
            self.name,
            "temporary upstream unavailable",
            cause_type="TimeoutError",
        )


class PermanentFailingProvider(MarketDataProvider):
    """Provider that fails with a terminal provider error."""

    name = "permanent-static"

    def fetch_daily_bars(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> ProviderFetchResult:
        raise ProviderPermanentError(
            self.name,
            "invalid provider response payload",
            cause_type="ValueError",
        )


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
    assert run.metadata_json["request"]["benchmark_symbol"] == "SPY"
    assert run.metadata_json["request"]["requested_symbols"] == ["AAPL"]
    assert run.metadata_json["request"]["fetch_symbols"] == ["AAPL", "SPY"]
    assert run.metadata_json["provider"]["name"] == "static"
    assert run.metadata_json["provider"]["config"] == {
        "max_attempts": 1,
        "backoff_seconds": 1.0,
        "backoff_multiplier": 2.0,
    }
    assert run.metadata_json["provider"]["metadata"] == {"fixture": "synthetic"}
    assert run.metadata_json["retry"] == {
        "configured_max_attempts": 1,
        "backoff_seconds": 1.0,
        "backoff_multiplier": 2.0,
        "attempt_count": 1,
        "completed_after_retry": False,
        "attempt_log": [{"attempt_number": 1, "status": "succeeded"}],
    }
    assert run.metadata_json["provider_fetch"]["returned_symbols"] == ["AAPL", "SPY"]
    assert run.metadata_json["provider_fetch"]["missing_symbols"] == []
    assert run.metadata_json["provider_fetch"]["bar_count"] == 180
    assert run.metadata_json["provider_fetch"]["per_symbol_bar_counts"] == {
        "AAPL": 90,
        "SPY": 90,
    }
    assert run.metadata_json["provider_fetch"]["first_trade_date"] == "2024-01-02"
    assert run.metadata_json["provider_fetch"]["last_trade_date"] == "2024-05-06"
    assert run.metadata_json["provider_fetch"]["warnings"] == []
    assert run.metadata_json["provider_fetch"]["source_updated_at_min"] is None
    assert run.metadata_json["provider_fetch"]["source_updated_at_max"] is None
    assert run.metadata_json["persistence"]["records_written"] == run.records_written

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


def test_ingestion_tracks_partial_provider_metadata(tmp_path: Path) -> None:
    """Ingestion metadata should make partial provider fetches explicit."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'partial.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
        default_horizons=[1, 5, 20],
    )
    create_all_tables(database_url)

    partial_bars = [bar for bar in build_synthetic_bars() if bar.symbol == "AAPL"]
    run = IngestionService(
        provider=StaticProvider(partial_bars),
        settings=settings,
    ).ingest_daily_bars(
        ["AAPL"],
        date(2024, 1, 2),
        date(2024, 5, 31),
    )

    assert run.status == "completed"
    assert run.records_written == 90
    assert run.metadata_json["request"]["fetch_symbols"] == ["AAPL", "SPY"]
    assert run.metadata_json["retry"]["attempt_count"] == 1
    assert run.metadata_json["provider_fetch"]["returned_symbols"] == ["AAPL"]
    assert run.metadata_json["provider_fetch"]["missing_symbols"] == ["SPY"]
    assert run.metadata_json["provider_fetch"]["per_symbol_bar_counts"] == {
        "AAPL": 90,
        "SPY": 0,
    }


def test_ingestion_persists_retry_metadata_for_transient_failures(tmp_path: Path) -> None:
    """Retry-aware metadata should record transient provider failures cleanly."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'transient.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
        default_horizons=[1, 5, 20],
    )
    create_all_tables(database_url)

    service = IngestionService(provider=TransientFailingProvider(), settings=settings)

    with pytest.raises(ProviderTransientError, match="temporary upstream unavailable"):
        service.ingest_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 5, 31))

    with session_scope(database_url) as session:
        run = session.execute(select(IngestionRun)).scalar_one()

    assert run.status == "failed"
    assert run.records_written == 0
    assert run.metadata_json["retry"] == {
        "configured_max_attempts": 1,
        "backoff_seconds": 1.0,
        "backoff_multiplier": 2.0,
        "attempt_count": 1,
        "completed_after_retry": False,
        "attempt_log": [
            {
                "attempt_number": 1,
                "status": "failed",
                "retriable": True,
                "error_type": "TimeoutError",
                "error_message": "temporary upstream unavailable",
            }
        ],
    }
    assert run.metadata_json["failure"] == {
        "error": "temporary upstream unavailable",
        "error_type": "TimeoutError",
        "retriable": True,
    }
    assert "provider_fetch" not in run.metadata_json


def test_ingestion_persists_retry_metadata_for_permanent_failures(tmp_path: Path) -> None:
    """Terminal provider failures should be marked non-retryable."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'permanent.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
        default_horizons=[1, 5, 20],
    )
    create_all_tables(database_url)

    service = IngestionService(provider=PermanentFailingProvider(), settings=settings)

    with pytest.raises(ProviderPermanentError, match="invalid provider response payload"):
        service.ingest_daily_bars(["AAPL"], date(2024, 1, 2), date(2024, 5, 31))

    with session_scope(database_url) as session:
        run = session.execute(select(IngestionRun)).scalar_one()

    assert run.status == "failed"
    assert run.records_written == 0
    assert run.metadata_json["retry"]["attempt_log"] == [
        {
            "attempt_number": 1,
            "status": "failed",
            "retriable": False,
            "error_type": "ValueError",
            "error_message": "invalid provider response payload",
        }
    ]
    assert run.metadata_json["failure"] == {
        "error": "invalid provider response payload",
        "error_type": "ValueError",
        "retriable": False,
    }
    assert "provider_fetch" not in run.metadata_json
