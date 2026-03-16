"""Integration test for backtesting and SHAP explainability."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd

from quant_signal.backtesting.service import BacktestService
from quant_signal.core.config import Settings
from quant_signal.explainability.service import ExplainabilityService
from quant_signal.features.pipeline import FeaturePipeline
from quant_signal.ingestion.models import MarketDataBar
from quant_signal.ingestion.providers import MarketDataProvider
from quant_signal.ingestion.service import IngestionService
from quant_signal.storage.db import create_all_tables, session_scope
from quant_signal.storage.repositories import StorageRepository
from quant_signal.training.service import TrainingService


class StaticProvider(MarketDataProvider):
    """Deterministic provider for backtest and SHAP tests."""

    name = "static"

    def __init__(self, periods: int) -> None:
        self._bars = self._build_bars(periods)

    def fetch_daily_bars(
        self,
        symbols: list[str] | tuple[str, ...],
        start_date: date,
        end_date: date,
    ) -> list[MarketDataBar]:
        requested = {symbol.upper() for symbol in symbols}
        return [
            bar
            for bar in self._bars
            if bar.symbol in requested and start_date <= bar.trade_date <= end_date
        ]

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


def test_backtesting_and_explainability(tmp_path: Path) -> None:
    """Backtest and SHAP runs should persist reproducible artifacts."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'backtest.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
        default_horizons=[1, 5, 20],
        min_training_days=80,
        top_n_signals=1,
    )
    create_all_tables(database_url)

    IngestionService(provider=StaticProvider(periods=220), settings=settings).ingest_daily_bars(
        ["AAPL"],
        date(2023, 1, 3),
        date(2023, 11, 30),
    )
    dataset = FeaturePipeline(settings=settings).build_dataset(date(2023, 11, 30), ["AAPL"])
    TrainingService(settings=settings).train(dataset.id)

    with session_scope(database_url) as session:
        repository = StorageRepository(session)
        champion_model = repository.list_champion_models(horizon_days=5)[0]

    backtest_run = BacktestService(settings=settings).run(champion_model.id, top_n=1)
    shap_run = ExplainabilityService(settings=settings).generate(
        champion_model.id,
        sample_size=8,
        top_signals=3,
    )

    assert Path(backtest_run.artifact_path).exists()
    assert Path(shap_run.artifact_path).exists()
    assert "cumulative_return" in backtest_run.summary_json
    assert shap_run.summary_json["global_importance"]
