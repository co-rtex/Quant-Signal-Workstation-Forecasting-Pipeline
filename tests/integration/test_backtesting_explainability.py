"""Integration test for backtesting and SHAP explainability."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quant_signal.backtesting.analytics import (
    BACKTEST_ANALYTICS_VERSION,
    BACKTEST_ARTIFACT_COLUMNS,
    BACKTEST_DETAIL_ARTIFACT_COLUMNS,
    BACKTEST_DETAIL_ARTIFACT_VERSION,
)
from quant_signal.backtesting.execution import BacktestExecutionAssumptions
from quant_signal.backtesting.regimes import REGIME_DEFINITION_VERSION
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

    backtest_service = BacktestService(settings=settings)
    baseline_backtest_run = backtest_service.run(champion_model.id, top_n=1)
    cost_backtest_run = backtest_service.run(
        champion_model.id,
        top_n=1,
        execution_assumptions=BacktestExecutionAssumptions(
            transaction_cost_bps=5.0,
            slippage_bps=2.0,
        ),
    )
    shap_run = ExplainabilityService(settings=settings).generate(
        champion_model.id,
        sample_size=8,
        top_signals=3,
    )

    baseline_artifact = pd.read_parquet(baseline_backtest_run.artifact_path)
    cost_artifact = pd.read_parquet(cost_backtest_run.artifact_path)
    detail_artifact = pd.read_parquet(cost_backtest_run.metadata_json["detail_artifact_path"])

    assert Path(baseline_backtest_run.artifact_path).exists()
    assert Path(cost_backtest_run.artifact_path).exists()
    assert Path(cost_backtest_run.metadata_json["detail_artifact_path"]).exists()
    assert Path(shap_run.artifact_path).exists()
    assert baseline_backtest_run.artifact_path != cost_backtest_run.artifact_path
    assert "cumulative_return" in baseline_backtest_run.summary_json
    assert shap_run.summary_json["global_importance"]
    assert list(cost_artifact.columns) == BACKTEST_ARTIFACT_COLUMNS
    assert list(detail_artifact.columns) == BACKTEST_DETAIL_ARTIFACT_COLUMNS
    assert np.allclose(
        baseline_artifact["gross_return"],
        baseline_artifact["net_return"],
    )
    assert np.allclose(
        baseline_artifact["net_return"],
        baseline_artifact["portfolio_return"],
    )
    assert np.allclose(
        cost_artifact["active_return"],
        cost_artifact["portfolio_return"] - cost_artifact["benchmark_return"],
    )
    assert np.allclose(
        cost_artifact["gross_active_return"],
        cost_artifact["gross_return"] - cost_artifact["benchmark_return"],
    )
    assert np.allclose(
        cost_artifact["portfolio_cumulative_return"],
        (1.0 + cost_artifact["portfolio_return"]).cumprod() - 1.0,
    )
    assert np.allclose(
        cost_artifact["benchmark_cumulative_return"],
        (1.0 + cost_artifact["benchmark_return"]).cumprod() - 1.0,
    )
    detail_reconciliation = (
        detail_artifact.groupby("active_date", as_index=False)
        .agg(
            gross_active_return=("gross_active_return_contribution", "sum"),
            active_return=("net_active_return_contribution", "sum"),
        )
        .rename(columns={"active_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    artifact_reconciliation = cost_artifact[
        ["date", "gross_active_return", "active_return"]
    ].sort_values("date").reset_index(drop=True)
    assert np.allclose(
        detail_reconciliation["gross_active_return"],
        artifact_reconciliation["gross_active_return"],
    )
    assert np.allclose(
        detail_reconciliation["active_return"],
        artifact_reconciliation["active_return"],
    )
    assert {"entries_count", "exits_count", "holdings_count", "turnover", "turnover_cost"}.issubset(
        cost_artifact.columns
    )
    assert (cost_artifact["turnover"] >= 0).all()
    assert (cost_artifact["holdings_count"] >= 0).all()
    assert (cost_artifact["transaction_cost"] > 0).any()
    assert (cost_artifact["slippage_cost"] > 0).any()
    assert np.all(cost_artifact["gross_return"] >= cost_artifact["net_return"])
    assert np.allclose(
        cost_artifact["net_return"],
        cost_artifact["portfolio_return"],
    )
    assert (
        cost_backtest_run.summary_json["gross_cumulative_return"]
        >= cost_backtest_run.summary_json["cumulative_return"]
    )
    assert cost_backtest_run.summary_json["total_transaction_cost"] > 0
    assert cost_backtest_run.summary_json["total_slippage_cost"] > 0
    assert cost_backtest_run.summary_json["total_cost_drag"] > 0
    assert "benchmark_metrics" in cost_backtest_run.summary_json
    assert "active_metrics" in cost_backtest_run.summary_json
    assert "turnover_metrics" in cost_backtest_run.summary_json
    assert "attribution_metrics" in cost_backtest_run.summary_json
    assert "lifecycle_attribution" in cost_backtest_run.summary_json
    assert "dimension_summaries" in cost_backtest_run.summary_json
    assert "attribution_dimension_summaries" in cost_backtest_run.summary_json
    assert cost_backtest_run.summary_json["benchmark_metrics"]["benchmark_symbol"] == "SPY"
    assert cost_backtest_run.summary_json["active_metrics"]["tracking_error"] >= 0
    assert cost_backtest_run.summary_json["turnover_metrics"]["average_turnover"] >= 0
    assert cost_backtest_run.summary_json["turnover_metrics"]["total_entries"] >= 0
    assert cost_backtest_run.summary_json["turnover_metrics"]["total_exits"] >= 0
    assert np.isclose(
        cost_backtest_run.summary_json["attribution_metrics"]["gross_active_return"],
        float(detail_artifact["gross_active_return_contribution"].sum()),
    )
    assert np.isclose(
        cost_backtest_run.summary_json["attribution_metrics"]["net_active_return"],
        float(detail_artifact["net_active_return_contribution"].sum()),
    )
    assert np.isclose(
        cost_backtest_run.summary_json["attribution_metrics"]["total_transaction_cost_drag"],
        float(detail_artifact["transaction_cost_contribution"].sum()),
    )
    assert np.isclose(
        cost_backtest_run.summary_json["attribution_metrics"]["total_slippage_cost_drag"],
        float(detail_artifact["slippage_cost_contribution"].sum()),
    )
    lifecycle_attribution = cost_backtest_run.summary_json["lifecycle_attribution"]
    assert {"entry", "held", "exit"} == set(lifecycle_attribution)
    assert lifecycle_attribution["entry"]["position_day_count"] >= 0
    assert lifecycle_attribution["held"]["position_day_count"] >= 0
    assert lifecycle_attribution["exit"]["position_day_count"] >= 0
    assert lifecycle_attribution["entry"]["transaction_cost_drag"] >= 0
    assert lifecycle_attribution["exit"]["slippage_cost_drag"] >= 0
    assert {
        "trend_flag",
        "volatility_flag",
        "momentum_flag",
        "drawdown_bucket",
    }.issubset(cost_backtest_run.summary_json["dimension_summaries"])
    assert {
        "trend_flag",
        "volatility_flag",
        "momentum_flag",
        "drawdown_bucket",
    }.issubset(cost_backtest_run.summary_json["attribution_dimension_summaries"])
    trend_summary = cost_backtest_run.summary_json["attribution_dimension_summaries"][
        "trend_flag"
    ]
    assert trend_summary
    trend_metrics = next(iter(trend_summary.values()))
    assert {
        "sample_count",
        "average_gross_active_return",
        "average_transaction_cost_drag",
        "average_slippage_cost_drag",
        "average_implementation_drag",
        "average_net_active_return",
        "active_hit_rate",
    }.issubset(trend_metrics)
    regime_metrics = next(iter(cost_backtest_run.regime_summary_json.values()))
    assert {
        "sample_count",
        "average_return",
        "average_gross_return",
        "benchmark_average_return",
        "average_active_return",
        "average_gross_active_return",
        "average_transaction_cost_drag",
        "average_slippage_cost_drag",
        "average_implementation_drag",
        "hit_rate",
        "active_hit_rate",
    }.issubset(regime_metrics)
    assert cost_backtest_run.metadata_json["execution_assumptions"] == {
        "transaction_cost_bps": 5.0,
        "slippage_bps": 2.0,
        "transaction_cost_rate": 0.0005,
        "slippage_rate": 0.0002,
        "total_cost_rate_per_side": 0.0007,
    }
    assert cost_backtest_run.metadata_json["sleeves_opened"] > 0
    assert cost_backtest_run.metadata_json["sleeves_closed"] > 0
    assert cost_backtest_run.metadata_json["benchmark_symbol"] == "SPY"
    assert (
        cost_backtest_run.metadata_json["backtest_detail_artifact_version"]
        == BACKTEST_DETAIL_ARTIFACT_VERSION
    )
    assert (
        cost_backtest_run.metadata_json["backtest_analytics_version"]
        == BACKTEST_ANALYTICS_VERSION
    )
    assert (
        cost_backtest_run.metadata_json["regime_definition_version"]
        == REGIME_DEFINITION_VERSION
    )
    assert cost_backtest_run.metadata_json["detail_artifact_hash"]
