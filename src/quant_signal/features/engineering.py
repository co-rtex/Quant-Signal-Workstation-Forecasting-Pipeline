"""Feature engineering for model-ready daily equity datasets."""

from __future__ import annotations

import math

import pandas as pd

FEATURE_COLUMNS = [
    "daily_return",
    "momentum_5d",
    "momentum_20d",
    "volatility_5d",
    "volatility_20d",
    "ma_spread_10_20",
    "price_vs_ma_20",
    "volume_change_1d",
    "volume_ratio_20d",
    "drawdown_20d",
    "relative_strength_5d",
    "relative_strength_20d",
    "benchmark_momentum_5d",
    "benchmark_momentum_20d",
    "benchmark_volatility_20d",
    "benchmark_trend",
    "benchmark_high_vol",
]


def _rolling_standard_deviation(series: pd.Series, window: int) -> pd.Series:
    """Return the rolling annualized standard deviation."""

    return series.rolling(window, min_periods=window).std() * math.sqrt(252)


def _build_benchmark_context(benchmark_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute benchmark-only context features."""

    benchmark = benchmark_frame.copy()
    benchmark["benchmark_daily_return"] = benchmark["adjusted_close"].pct_change()
    benchmark["benchmark_momentum_5d"] = benchmark["adjusted_close"].pct_change(5)
    benchmark["benchmark_momentum_20d"] = benchmark["adjusted_close"].pct_change(20)
    benchmark["benchmark_volatility_20d"] = _rolling_standard_deviation(
        benchmark["benchmark_daily_return"],
        20,
    )
    benchmark["benchmark_ma_50"] = benchmark["adjusted_close"].rolling(50, min_periods=50).mean()
    benchmark["benchmark_trend"] = (
        benchmark["adjusted_close"] > benchmark["benchmark_ma_50"]
    ).astype(int)
    benchmark["benchmark_volatility_median_60d"] = benchmark["benchmark_volatility_20d"].rolling(
        60,
        min_periods=20,
    ).median()
    benchmark["benchmark_high_vol"] = (
        benchmark["benchmark_volatility_20d"] > benchmark["benchmark_volatility_median_60d"]
    ).astype(int)
    return benchmark[
        [
            "date",
            "benchmark_momentum_5d",
            "benchmark_momentum_20d",
            "benchmark_volatility_20d",
            "benchmark_trend",
            "benchmark_high_vol",
        ]
    ]


def build_feature_frame(bars: pd.DataFrame, benchmark_symbol: str) -> pd.DataFrame:
    """Construct a leakage-safe feature frame from normalized daily bars."""

    if bars.empty:
        raise ValueError("Cannot build features from an empty bar set.")

    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

    grouped = frame.groupby("symbol", sort=False)
    frame["daily_return"] = grouped["adjusted_close"].pct_change()
    frame["momentum_5d"] = grouped["adjusted_close"].pct_change(5)
    frame["momentum_20d"] = grouped["adjusted_close"].pct_change(20)
    frame["volatility_5d"] = grouped["daily_return"].transform(
        lambda series: _rolling_standard_deviation(series, 5)
    )
    frame["volatility_20d"] = grouped["daily_return"].transform(
        lambda series: _rolling_standard_deviation(series, 20)
    )
    frame["ma_10"] = grouped["adjusted_close"].transform(
        lambda series: series.rolling(10, min_periods=10).mean()
    )
    frame["ma_20"] = grouped["adjusted_close"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    frame["ma_spread_10_20"] = (frame["ma_10"] / frame["ma_20"]) - 1.0
    frame["price_vs_ma_20"] = (frame["adjusted_close"] / frame["ma_20"]) - 1.0
    frame["volume_change_1d"] = grouped["volume"].pct_change()
    frame["volume_mean_20d"] = grouped["volume"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    frame["volume_ratio_20d"] = frame["volume"] / frame["volume_mean_20d"]
    frame["rolling_max_20d"] = grouped["adjusted_close"].transform(
        lambda series: series.rolling(20, min_periods=20).max()
    )
    frame["drawdown_20d"] = (frame["adjusted_close"] / frame["rolling_max_20d"]) - 1.0

    benchmark_rows = frame[frame["symbol"] == benchmark_symbol.upper()].copy()
    if benchmark_rows.empty:
        raise ValueError(f"Benchmark symbol {benchmark_symbol} is missing from the bar set.")

    benchmark_context = _build_benchmark_context(benchmark_rows)
    enriched = frame.merge(benchmark_context, on="date", how="left")
    enriched["relative_strength_5d"] = (
        enriched["momentum_5d"] - enriched["benchmark_momentum_5d"]
    )
    enriched["relative_strength_20d"] = (
        enriched["momentum_20d"] - enriched["benchmark_momentum_20d"]
    )

    return enriched.drop(columns=["ma_10", "ma_20", "volume_mean_20d", "rolling_max_20d"])
