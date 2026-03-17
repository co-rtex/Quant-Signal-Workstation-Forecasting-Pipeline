"""Tests for benchmark-relative backtest analytics."""

from __future__ import annotations

import math

import pandas as pd

from quant_signal.backtesting.analytics import (
    BACKTEST_DETAIL_ARTIFACT_COLUMNS,
    BACKTEST_DETAIL_BASE_COLUMNS,
    attach_benchmark_relative_analytics,
    attach_detail_benchmark_attribution,
    build_attribution_metrics,
    build_benchmark_relative_summary,
    build_lifecycle_attribution,
    build_turnover_daily_metrics,
    build_turnover_summary,
)
from quant_signal.backtesting.execution import BacktestExecutionAssumptions


def test_attach_benchmark_relative_analytics_computes_active_returns() -> None:
    """Benchmark-relative analytics should align with portfolio and benchmark returns."""

    portfolio_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "gross_return": [0.0100, 0.0200, -0.0100],
            "transaction_cost": [0.0000, 0.0000, 0.0000],
            "slippage_cost": [0.0000, 0.0000, 0.0000],
            "net_return": [0.0100, 0.0200, -0.0100],
            "active_sleeves": [1, 1, 1],
            "entries_count": [1, 1, 1],
            "exits_count": [0, 0, 0],
            "holdings_count": [1, 1, 1],
            "turnover": [1.0, 0.0, 0.0],
            "turnover_cost": [0.0, 0.0, 0.0],
            "portfolio_return": [0.0100, 0.0200, -0.0100],
        }
    )
    benchmark_analytics = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "benchmark_return": [0.0050, 0.0100, -0.0020],
            "regime": ["bull_low_vol", "bull_low_vol", "bull_low_vol"],
            "trend_flag": ["bull", "bull", "bull"],
            "volatility_flag": ["low_vol", "low_vol", "low_vol"],
            "momentum_flag": [
                "positive_momentum",
                "positive_momentum",
                "positive_momentum",
            ],
            "drawdown_bucket": ["at_high", "at_high", "pullback"],
            "volatility_20d": [0.10, 0.10, 0.10],
            "momentum_20d": [0.03, 0.03, 0.01],
            "drawdown_63d": [0.0, -0.01, -0.03],
        }
    )

    analytics = attach_benchmark_relative_analytics(
        portfolio_returns,
        benchmark_analytics,
    )

    expected_relative = (
        ((1.0 + portfolio_returns["portfolio_return"]).cumprod())
        / ((1.0 + benchmark_analytics["benchmark_return"]).cumprod())
        - 1.0
    )

    assert analytics["active_return"].tolist() == [0.0050, 0.0100, -0.0080]
    assert analytics["gross_active_return"].tolist() == [0.0050, 0.0100, -0.0080]
    assert analytics["relative_cumulative_return"].round(10).tolist() == (
        expected_relative.round(10).tolist()
    )


def test_build_benchmark_relative_summary_returns_tracking_error() -> None:
    """Benchmark summary helpers should expose active-return metrics."""

    analytics_frame = pd.DataFrame(
        {
            "portfolio_return": [0.0100, 0.0200, -0.0100],
            "gross_return": [0.0110, 0.0210, -0.0090],
            "benchmark_return": [0.0050, 0.0100, -0.0020],
            "active_return": [0.0050, 0.0100, -0.0080],
            "gross_active_return": [0.0060, 0.0110, -0.0070],
            "relative_cumulative_return": [0.0049751244, 0.0148768473, 0.0068641180],
            "benchmark_drawdown": [0.0, 0.0, -0.0020],
            "relative_drawdown": [0.0, 0.0, -0.0078941176],
        }
    )

    summary = build_benchmark_relative_summary(analytics_frame, benchmark_symbol="SPY")
    expected_tracking_error = float(
        pd.Series([0.0050, 0.0100, -0.0080]).std(ddof=0) * math.sqrt(252)
    )

    assert summary["benchmark_metrics"]["benchmark_symbol"] == "SPY"
    assert summary["active_metrics"]["average_active_return"] == float(
        analytics_frame["active_return"].mean()
    )
    assert summary["active_metrics"]["tracking_error"] == expected_tracking_error
    assert summary["active_metrics"]["information_ratio"] is not None
    assert summary["active_metrics"]["relative_max_drawdown"] == -0.0078941176


def test_build_turnover_daily_metrics_tracks_entries_and_exits() -> None:
    """Turnover metrics should reflect composition changes across active dates."""

    detail_frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "active_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "symbol": ["AAPL", "MSFT", "AAPL", "NVDA"],
            "rank": [1, 2, 1, 2],
            "weight": [0.5, 0.5, 0.5, 0.5],
            "is_entry": [True, True, False, True],
            "is_exit": [False, False, False, False],
            "is_held": [False, False, True, False],
            "gross_return_contribution": [0.01, 0.00, 0.02, -0.01],
            "transaction_cost_contribution": [0.0, 0.0, 0.0, 0.0],
            "slippage_cost_contribution": [0.0, 0.0, 0.0, 0.0],
            "net_return_contribution": [0.01, 0.00, 0.02, -0.01],
        }
    )[BACKTEST_DETAIL_BASE_COLUMNS]

    turnover = build_turnover_daily_metrics(
        detail_frame,
        BacktestExecutionAssumptions(transaction_cost_bps=5.0, slippage_bps=2.0),
    )

    assert turnover["entries_count"].tolist() == [2, 1]
    assert turnover["exits_count"].tolist() == [0, 1]
    assert turnover["holdings_count"].tolist() == [2, 2]
    assert turnover["turnover"].tolist() == [1.0, 0.5]
    assert turnover["turnover_cost"].tolist() == [0.0007, 0.00035]


def test_build_turnover_summary_returns_non_negative_metrics() -> None:
    """Turnover summary should expose aggregate turnover diagnostics."""

    analytics_frame = pd.DataFrame(
        {
            "transaction_cost": [0.0004, 0.0002],
            "slippage_cost": [0.0002, 0.0001],
            "entries_count": [2, 1],
            "exits_count": [0, 1],
            "holdings_count": [2, 2],
            "turnover": [1.0, 0.5],
            "turnover_cost": [0.0007, 0.00035],
        }
    )

    summary = build_turnover_summary(analytics_frame)

    assert summary["average_turnover"] == 0.75
    assert summary["max_turnover"] == 1.0
    assert summary["average_holdings_count"] == 2.0
    assert summary["total_entries"] == 3
    assert summary["total_exits"] == 1
    assert summary["turnover_cost_share"] > 0


def test_attach_detail_benchmark_attribution_computes_position_level_contributions() -> None:
    """Detail attribution should attach benchmark-relative contribution fields."""

    detail_frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "active_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "symbol": ["AAPL", "AAPL"],
            "rank": [1, 1],
            "weight": [1.0, 1.0],
            "is_entry": [True, False],
            "is_exit": [False, True],
            "is_held": [False, False],
            "gross_return_contribution": [0.0100, 0.0200],
            "transaction_cost_contribution": [0.0005, 0.0005],
            "slippage_cost_contribution": [0.0002, 0.0002],
            "net_return_contribution": [0.0093, 0.0193],
        }
    )[BACKTEST_DETAIL_BASE_COLUMNS]
    benchmark_analytics = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "benchmark_return": [0.0050, 0.0100],
        }
    )

    attributed_detail = attach_detail_benchmark_attribution(
        detail_frame,
        benchmark_analytics,
    )

    assert list(attributed_detail.columns) == BACKTEST_DETAIL_ARTIFACT_COLUMNS
    assert attributed_detail["benchmark_return"].tolist() == [0.0050, 0.0100]
    assert attributed_detail["benchmark_return_contribution"].tolist() == [0.0050, 0.0100]
    assert attributed_detail["gross_active_return_contribution"].tolist() == [0.0050, 0.0100]
    assert attributed_detail["implementation_drag_contribution"].tolist() == [0.0007, 0.0007]
    assert math.isclose(
        attributed_detail["net_active_return_contribution"].iloc[0],
        0.0043,
    )
    assert math.isclose(
        attributed_detail["net_active_return_contribution"].iloc[1],
        0.0093,
    )


def test_build_attribution_metrics_and_lifecycle_summaries() -> None:
    """Attribution helpers should return additive totals and lifecycle slices."""

    detail_frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01"]
            ),
            "active_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "rank": [1, 1, 1],
            "weight": [1.0, 1.0, 1.0],
            "is_entry": [True, False, False],
            "is_exit": [False, False, True],
            "is_held": [False, True, False],
            "gross_return_contribution": [0.0100, 0.0200, -0.0100],
            "transaction_cost_contribution": [0.0005, 0.0, 0.0005],
            "slippage_cost_contribution": [0.0002, 0.0, 0.0002],
            "net_return_contribution": [0.0093, 0.0200, -0.0107],
        }
    )[BACKTEST_DETAIL_BASE_COLUMNS]
    benchmark_analytics = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "benchmark_return": [0.0050, 0.0100, -0.0020],
        }
    )

    attributed_detail = attach_detail_benchmark_attribution(
        detail_frame,
        benchmark_analytics,
    )
    attribution_metrics = build_attribution_metrics(attributed_detail)
    lifecycle_attribution = build_lifecycle_attribution(attributed_detail)

    assert math.isclose(attribution_metrics["gross_active_return"], 0.007)
    assert math.isclose(attribution_metrics["total_transaction_cost_drag"], 0.001)
    assert math.isclose(attribution_metrics["total_slippage_cost_drag"], 0.0004)
    assert math.isclose(attribution_metrics["total_implementation_drag"], 0.0014)
    assert math.isclose(attribution_metrics["net_active_return"], 0.0056)
    assert lifecycle_attribution["entry"]["position_day_count"] == 1
    assert lifecycle_attribution["held"]["position_day_count"] == 1
    assert lifecycle_attribution["exit"]["position_day_count"] == 1
    assert math.isclose(lifecycle_attribution["entry"]["average_weight"], 1.0)
    assert math.isclose(lifecycle_attribution["held"]["transaction_cost_drag"], 0.0)
    assert math.isclose(lifecycle_attribution["exit"]["slippage_cost_drag"], 0.0002)
    assert math.isclose(
        lifecycle_attribution["exit"]["net_active_return_contribution"],
        -0.0087,
    )
