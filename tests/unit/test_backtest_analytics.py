"""Tests for benchmark-relative backtest analytics."""

from __future__ import annotations

import math

import pandas as pd

from quant_signal.backtesting.analytics import (
    attach_benchmark_relative_analytics,
    build_benchmark_relative_summary,
)


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
