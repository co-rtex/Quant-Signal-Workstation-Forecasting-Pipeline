"""Benchmark-relative analytics helpers for backtests."""

from __future__ import annotations

import math

import pandas as pd

BACKTEST_ANALYTICS_VERSION = "v2"
REGIME_DIMENSIONS = (
    "trend_flag",
    "volatility_flag",
    "momentum_flag",
    "drawdown_bucket",
)
BACKTEST_ARTIFACT_COLUMNS = [
    "date",
    "gross_return",
    "transaction_cost",
    "slippage_cost",
    "net_return",
    "active_sleeves",
    "portfolio_return",
    "benchmark_return",
    "gross_active_return",
    "active_return",
    "portfolio_cumulative_return",
    "benchmark_cumulative_return",
    "relative_cumulative_return",
    "portfolio_drawdown",
    "benchmark_drawdown",
    "relative_drawdown",
    "regime",
    "trend_flag",
    "volatility_flag",
    "momentum_flag",
    "drawdown_bucket",
    "volatility_20d",
    "momentum_20d",
    "drawdown_63d",
]


def compute_equity_curve(returns: pd.Series) -> pd.Series:
    """Return the cumulative equity curve for a return series."""

    return (1.0 + returns.astype(float)).cumprod()


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Return the drawdown series for an equity curve."""

    if equity_curve.empty:
        return pd.Series(dtype=float)
    return equity_curve / equity_curve.cummax() - 1.0


def annualized_return_from_equity(equity_curve: pd.Series) -> float:
    """Return the annualized return for an equity curve."""

    if equity_curve.empty:
        return 0.0
    return float(equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1.0)


def attach_benchmark_relative_analytics(
    portfolio_returns: pd.DataFrame,
    benchmark_analytics: pd.DataFrame,
) -> pd.DataFrame:
    """Attach benchmark-relative analytics and regime context to backtest rows."""

    if portfolio_returns.empty:
        return pd.DataFrame(columns=BACKTEST_ARTIFACT_COLUMNS)

    benchmark_columns = [
        "date",
        "benchmark_return",
        "regime",
        "trend_flag",
        "volatility_flag",
        "momentum_flag",
        "drawdown_bucket",
        "volatility_20d",
        "momentum_20d",
        "drawdown_63d",
    ]
    analytics_frame = portfolio_returns.merge(
        benchmark_analytics[benchmark_columns],
        on="date",
        how="left",
    )

    missing_dates = (
        analytics_frame.loc[analytics_frame["benchmark_return"].isna(), "date"]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    if missing_dates:
        preview = ", ".join(missing_dates[:5])
        raise ValueError(
            f"Missing benchmark coverage for backtest dates: {preview}"
        )

    analytics_frame["gross_active_return"] = (
        analytics_frame["gross_return"] - analytics_frame["benchmark_return"]
    )
    analytics_frame["active_return"] = (
        analytics_frame["portfolio_return"] - analytics_frame["benchmark_return"]
    )

    portfolio_equity = compute_equity_curve(analytics_frame["portfolio_return"])
    benchmark_equity = compute_equity_curve(analytics_frame["benchmark_return"])
    relative_equity = portfolio_equity / benchmark_equity

    analytics_frame["portfolio_cumulative_return"] = portfolio_equity - 1.0
    analytics_frame["benchmark_cumulative_return"] = benchmark_equity - 1.0
    analytics_frame["relative_cumulative_return"] = relative_equity - 1.0
    analytics_frame["portfolio_drawdown"] = compute_drawdown(portfolio_equity)
    analytics_frame["benchmark_drawdown"] = compute_drawdown(benchmark_equity)
    analytics_frame["relative_drawdown"] = compute_drawdown(relative_equity)
    return analytics_frame[BACKTEST_ARTIFACT_COLUMNS].copy()


def build_group_summary(
    analytics_frame: pd.DataFrame,
    group_column: str,
) -> dict[str, dict[str, object]]:
    """Build a standardized grouped performance summary."""

    if analytics_frame.empty or group_column not in analytics_frame.columns:
        return {}

    grouped = analytics_frame.dropna(subset=[group_column]).groupby(group_column)
    summary: dict[str, dict[str, object]] = {}
    for group_name, frame in grouped:
        summary[str(group_name)] = {
            "sample_count": int(len(frame)),
            "average_return": float(frame["portfolio_return"].mean()),
            "average_gross_return": float(frame["gross_return"].mean()),
            "benchmark_average_return": float(frame["benchmark_return"].mean()),
            "average_active_return": float(frame["active_return"].mean()),
            "average_gross_active_return": float(frame["gross_active_return"].mean()),
            "hit_rate": float((frame["portfolio_return"] > 0).mean()),
            "active_hit_rate": float((frame["active_return"] > 0).mean()),
        }
    return summary


def build_benchmark_relative_summary(
    analytics_frame: pd.DataFrame,
    benchmark_symbol: str,
) -> dict[str, dict[str, object]]:
    """Build summary sections for benchmark-relative portfolio performance."""

    if analytics_frame.empty:
        return {
            "benchmark_metrics": {
                "benchmark_symbol": benchmark_symbol,
                "benchmark_cumulative_return": 0.0,
                "benchmark_annualized_return": 0.0,
                "benchmark_annualized_volatility": 0.0,
                "benchmark_max_drawdown": 0.0,
            },
            "active_metrics": {
                "average_active_return": 0.0,
                "gross_average_active_return": 0.0,
                "active_hit_rate": 0.0,
                "relative_cumulative_return": 0.0,
                "tracking_error": 0.0,
                "information_ratio": None,
                "relative_max_drawdown": 0.0,
            },
        }

    benchmark_returns = analytics_frame["benchmark_return"].astype(float)
    active_returns = analytics_frame["active_return"].astype(float)
    benchmark_equity = compute_equity_curve(benchmark_returns)
    relative_equity = analytics_frame["relative_cumulative_return"].astype(float) + 1.0

    benchmark_annualized_return = annualized_return_from_equity(benchmark_equity)
    relative_annualized_return = annualized_return_from_equity(relative_equity)
    tracking_error = float(active_returns.std(ddof=0) * math.sqrt(252))
    information_ratio = (
        float(relative_annualized_return / tracking_error)
        if tracking_error > 0
        else None
    )

    return {
        "benchmark_metrics": {
            "benchmark_symbol": benchmark_symbol,
            "benchmark_cumulative_return": float(benchmark_equity.iloc[-1] - 1.0),
            "benchmark_annualized_return": benchmark_annualized_return,
            "benchmark_annualized_volatility": float(
                benchmark_returns.std(ddof=0) * math.sqrt(252)
            ),
            "benchmark_max_drawdown": float(analytics_frame["benchmark_drawdown"].min()),
        },
        "active_metrics": {
            "average_active_return": float(active_returns.mean()),
            "gross_average_active_return": float(
                analytics_frame["gross_active_return"].mean()
            ),
            "active_hit_rate": float((active_returns > 0).mean()),
            "relative_cumulative_return": float(
                analytics_frame["relative_cumulative_return"].iloc[-1]
            ),
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "relative_max_drawdown": float(analytics_frame["relative_drawdown"].min()),
        },
    }


def build_dimension_summaries(
    analytics_frame: pd.DataFrame,
) -> dict[str, dict[str, dict[str, object]]]:
    """Build grouped summaries for each supported regime dimension."""

    return {
        dimension: build_group_summary(analytics_frame, dimension)
        for dimension in REGIME_DIMENSIONS
    }
