"""Benchmark-relative analytics helpers for backtests."""

from __future__ import annotations

import math

import pandas as pd

from quant_signal.backtesting.execution import BacktestExecutionAssumptions

BACKTEST_ANALYTICS_VERSION = "v3"
BACKTEST_DETAIL_ARTIFACT_VERSION = "v1"
REGIME_DIMENSIONS = (
    "trend_flag",
    "volatility_flag",
    "momentum_flag",
    "drawdown_bucket",
)
BACKTEST_TURNOVER_COLUMNS = [
    "entries_count",
    "exits_count",
    "holdings_count",
    "turnover",
    "turnover_cost",
]
BACKTEST_ARTIFACT_COLUMNS = [
    "date",
    "gross_return",
    "transaction_cost",
    "slippage_cost",
    "net_return",
    "active_sleeves",
    *BACKTEST_TURNOVER_COLUMNS,
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
BACKTEST_DETAIL_ARTIFACT_COLUMNS = [
    "signal_date",
    "active_date",
    "symbol",
    "rank",
    "weight",
    "is_entry",
    "is_exit",
    "is_held",
    "gross_return_contribution",
    "transaction_cost_contribution",
    "slippage_cost_contribution",
    "net_return_contribution",
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

    portfolio_frame = portfolio_returns.copy()
    default_values: dict[str, int | float] = {
        "entries_count": 0,
        "exits_count": 0,
        "holdings_count": 0,
        "turnover": 0.0,
        "turnover_cost": 0.0,
    }
    for column, default in default_values.items():
        if column not in portfolio_frame.columns:
            portfolio_frame[column] = default

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
    analytics_frame = portfolio_frame.merge(
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


def build_turnover_daily_metrics(
    detail_frame: pd.DataFrame,
    execution_assumptions: BacktestExecutionAssumptions,
) -> pd.DataFrame:
    """Build daily turnover metrics from composition-level detail rows."""

    if detail_frame.empty:
        return pd.DataFrame(columns=["date", *BACKTEST_TURNOVER_COLUMNS])

    epsilon = 1e-12
    composition = (
        detail_frame.groupby(["active_date", "symbol"], as_index=False)
        .agg(weight=("weight", "sum"))
        .sort_values(["active_date", "symbol"])
    )

    rows: list[dict[str, object]] = []
    previous_weights = pd.Series(dtype=float)
    for active_date, frame in composition.groupby("active_date", sort=True):
        current_weights = frame.set_index("symbol")["weight"].sort_index()
        union_index = previous_weights.index.union(current_weights.index)
        previous = previous_weights.reindex(union_index, fill_value=0.0)
        current = current_weights.reindex(union_index, fill_value=0.0)
        deltas = current - previous
        bought_weight = float(deltas.clip(lower=0.0).sum())
        sold_weight = float((-deltas.clip(upper=0.0)).sum())
        turnover = float(max(bought_weight, sold_weight))
        rows.append(
            {
                "date": pd.Timestamp(active_date),
                "entries_count": int(
                    ((previous <= epsilon) & (current > epsilon)).sum()
                ),
                "exits_count": int(
                    ((previous > epsilon) & (current <= epsilon)).sum()
                ),
                "holdings_count": int((current > epsilon).sum()),
                "turnover": turnover,
                "turnover_cost": float(
                    turnover * execution_assumptions.total_cost_rate_per_side
                ),
            }
        )
        previous_weights = current_weights

    return pd.DataFrame(rows)


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


def build_turnover_summary(portfolio_returns: pd.DataFrame) -> dict[str, object]:
    """Build summary statistics for turnover-aware backtest reporting."""

    if portfolio_returns.empty:
        return {
            "average_turnover": 0.0,
            "max_turnover": 0.0,
            "average_holdings_count": 0.0,
            "total_entries": 0,
            "total_exits": 0,
            "turnover_cost_share": 0.0,
        }

    total_cost_drag = float(
        portfolio_returns["transaction_cost"].sum() + portfolio_returns["slippage_cost"].sum()
    )
    total_turnover_cost = float(portfolio_returns["turnover_cost"].sum())
    turnover_cost_share = (
        float(total_turnover_cost / total_cost_drag)
        if total_cost_drag > 0
        else 0.0
    )

    return {
        "average_turnover": float(portfolio_returns["turnover"].mean()),
        "max_turnover": float(portfolio_returns["turnover"].max()),
        "average_holdings_count": float(portfolio_returns["holdings_count"].mean()),
        "total_entries": int(portfolio_returns["entries_count"].sum()),
        "total_exits": int(portfolio_returns["exits_count"].sum()),
        "turnover_cost_share": turnover_cost_share,
    }


def build_dimension_summaries(
    analytics_frame: pd.DataFrame,
) -> dict[str, dict[str, dict[str, object]]]:
    """Build grouped summaries for each supported regime dimension."""

    return {
        dimension: build_group_summary(analytics_frame, dimension)
        for dimension in REGIME_DIMENSIONS
    }
