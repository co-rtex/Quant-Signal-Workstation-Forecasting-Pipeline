"""Integration test for training and read-only API serving."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from quant_signal.api.app import create_app
from quant_signal.core.config import Settings
from quant_signal.features.pipeline import FeaturePipeline
from quant_signal.ingestion.models import MarketDataBar, ProviderFetchResult
from quant_signal.ingestion.providers import MarketDataProvider
from quant_signal.ingestion.service import IngestionService
from quant_signal.storage.db import create_all_tables, session_scope
from quant_signal.storage.repositories import StorageRepository
from quant_signal.training.service import TrainingService


class StaticProvider(MarketDataProvider):
    """Deterministic provider for API training tests."""

    name = "static"

    def __init__(self, periods: int) -> None:
        self._bars = self._build_bars(periods)

    def fetch_daily_bars(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> ProviderFetchResult:
        requested = {symbol.upper() for symbol in symbols}
        bars = [
            bar
            for bar in self._bars
            if bar.symbol in requested and start_date <= bar.trade_date <= end_date
        ]
        return ProviderFetchResult.from_bars(
            bars,
            provider_metadata={"fixture": "synthetic"},
        )

    def _build_bars(self, periods: int) -> list[MarketDataBar]:
        bars: list[MarketDataBar] = []
        dates = pd.bdate_range("2023-01-03", periods=periods)
        for index, timestamp in enumerate(dates):
            trade_date = timestamp.date()
            aapl_close = 95.0 + (index * 0.18) + (math.sin(index / 5.0) * 7.5)
            spy_close = 390.0 + (index * 0.1) + (math.sin(index / 8.0) * 4.0)
            bars.append(
                MarketDataBar(
                    symbol="AAPL",
                    trade_date=trade_date,
                    open=aapl_close - 0.5,
                    high=aapl_close + 0.9,
                    low=aapl_close - 1.1,
                    close=aapl_close,
                    adjusted_close=aapl_close,
                    volume=1_500_000 + (index * 500),
                )
            )
            bars.append(
                MarketDataBar(
                    symbol="SPY",
                    trade_date=trade_date,
                    open=spy_close - 0.6,
                    high=spy_close + 0.8,
                    low=spy_close - 1.0,
                    close=spy_close,
                    adjusted_close=spy_close,
                    volume=6_000_000 + (index * 1_000),
                )
            )
        return bars


def test_training_and_api_endpoints(tmp_path: Path, monkeypatch: object) -> None:
    """Training should persist model outputs that the API can serve."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'training.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
        default_horizons=[1, 5, 20],
        min_training_days=80,
    )
    create_all_tables(database_url)

    provider = StaticProvider(periods=220)
    IngestionService(provider=provider, settings=settings).ingest_daily_bars(
        ["AAPL"],
        date(2023, 1, 3),
        date(2023, 11, 30),
    )
    dataset = FeaturePipeline(settings=settings).build_dataset(date(2023, 11, 30), ["AAPL"])
    trained_models = TrainingService(settings=settings).train(dataset.id)

    assert len(trained_models) == 6

    with session_scope(database_url) as session:
        repository = StorageRepository(session)
        champion_model = repository.list_champion_models(horizon_days=5)[0]
    latest_signal_date = str(dataset.metadata_json["date_range"]["end"])

    monkeypatch.setenv("DATABASE_URL", database_url)
    app = create_app()
    client = TestClient(app)

    signals_response = client.get(
        "/v1/signals",
        params={"as_of_date": latest_signal_date, "horizon": 5, "limit": 1},
    )
    model_response = client.get(f"/v1/models/{champion_model.id}")
    evaluation_response = client.get(f"/v1/models/{champion_model.id}/evaluation")

    assert signals_response.status_code == 200
    assert model_response.status_code == 200
    assert evaluation_response.status_code == 200
    assert signals_response.json()["signals"]
    assert model_response.json()["model_version_id"] == champion_model.id
    assert len(evaluation_response.json()["evaluations"]) == 2
