"""Integration tests for the pipeline CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from quant_signal.cli.pipeline import ServiceFactories, main
from quant_signal.core.config import Settings
from quant_signal.ingestion.errors import ProviderPermanentError
from quant_signal.ingestion.models import MarketDataBar, ProviderFetchResult
from quant_signal.ingestion.providers import MarketDataProvider
from quant_signal.ingestion.service import IngestionService
from quant_signal.storage.db import create_all_tables, session_scope
from quant_signal.storage.models import IngestionRun


class StaticProvider(MarketDataProvider):
    """Deterministic provider for CLI integration tests."""

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


def test_pipeline_ingest_command_persists_run_and_emits_json_summary(tmp_path: Path) -> None:
    """The ingest command should print a machine-readable success summary."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
    )
    create_all_tables(database_url)

    stdout = StringIO()
    exit_code = main(
        [
            "ingest",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-05-31",
            "--symbols",
            "AAPL",
        ],
        settings=settings,
        stdout=stdout,
        service_factories=ServiceFactories(
            ingestion=lambda resolved_settings: IngestionService(
                provider=StaticProvider(build_synthetic_bars()),
                settings=resolved_settings,
                sleep_fn=lambda _: None,
            )
        ),
    )

    assert exit_code == 0

    payload = json.loads(stdout.getvalue())
    assert payload == {
        "benchmark_symbol": "SPY",
        "command": "ingest",
        "completed_after_retry": False,
        "end_date": "2024-05-31",
        "fetch_symbols": ["AAPL", "SPY"],
        "provider": "static",
        "records_written": 180,
        "requested_symbols": ["AAPL"],
        "retry_attempt_count": 1,
        "run_id": payload["run_id"],
        "start_date": "2024-01-02",
        "status": "completed",
    }

    with session_scope(database_url) as session:
        run = session.scalar(select(IngestionRun).where(IngestionRun.id == payload["run_id"]))

    assert run is not None
    assert run.status == "completed"
    assert run.records_written == 180


def test_pipeline_ingest_command_defaults_to_settings_universe(tmp_path: Path) -> None:
    """The ingest command should use configured universe symbols when none are provided."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-default.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
    )
    create_all_tables(database_url)

    stdout = StringIO()
    exit_code = main(
        [
            "ingest",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-05-31",
        ],
        settings=settings,
        stdout=stdout,
        service_factories=ServiceFactories(
            ingestion=lambda resolved_settings: IngestionService(
                provider=StaticProvider(build_synthetic_bars()),
                settings=resolved_settings,
                sleep_fn=lambda _: None,
            )
        ),
    )

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["requested_symbols"] == ["AAPL", "SPY"]
    assert payload["fetch_symbols"] == ["AAPL", "SPY"]
    assert payload["records_written"] == 180


def test_pipeline_ingest_command_returns_nonzero_and_logs_failures(
    tmp_path: Path,
) -> None:
    """The ingest command should return a non-zero exit code on service failures."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-failed.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
    )
    create_all_tables(database_url)

    stdout = StringIO()
    stderr = StringIO()
    exit_code = main(
        [
            "ingest",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-05-31",
            "--symbols",
            "AAPL",
        ],
        settings=settings,
        stdout=stdout,
        stderr=stderr,
        service_factories=ServiceFactories(
            ingestion=lambda resolved_settings: IngestionService(
                provider=PermanentFailingProvider(),
                settings=resolved_settings,
                sleep_fn=lambda _: None,
            )
        ),
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert json.loads(stderr.getvalue()) == {
        "command": "ingest",
        "error": "invalid provider response payload",
        "error_type": "ProviderPermanentError",
        "status": "failed",
    }

    with session_scope(database_url) as session:
        run = session.scalar(select(IngestionRun).order_by(IngestionRun.started_at.desc()))

    assert run is not None
    assert run.status == "failed"
    assert run.metadata_json["retry"]["attempt_count"] == 1
    assert run.metadata_json["failure"] == {
        "error": "invalid provider response payload",
        "error_type": "ValueError",
        "retriable": False,
    }
