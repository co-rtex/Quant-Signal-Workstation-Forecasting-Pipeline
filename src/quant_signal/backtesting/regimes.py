"""Market regime labeling utilities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

REGIME_DEFINITION_VERSION = "compat_2x2_plus_v2"


def label_regimes(benchmark_frame: pd.DataFrame) -> pd.DataFrame:
    """Label each benchmark date with a simple trend/volatility regime."""

    frame = benchmark_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["benchmark_return"] = frame["adjusted_close"].pct_change()
    frame["trend_ma_50"] = frame["adjusted_close"].rolling(50, min_periods=50).mean()
    frame["volatility_20d"] = (
        frame["benchmark_return"].rolling(20, min_periods=20).std() * math.sqrt(252)
    )
    frame["volatility_median_60d"] = frame["volatility_20d"].rolling(60, min_periods=20).median()
    frame["momentum_20d"] = frame["adjusted_close"].pct_change(20)
    frame["rolling_peak_63d"] = frame["adjusted_close"].rolling(63, min_periods=1).max()
    frame["drawdown_63d"] = frame["adjusted_close"] / frame["rolling_peak_63d"] - 1.0

    frame["trend_flag"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame.loc[
        frame["trend_ma_50"].notna(),
        "trend_flag",
    ] = np.where(
        frame.loc[frame["trend_ma_50"].notna(), "adjusted_close"].ge(
            frame.loc[frame["trend_ma_50"].notna(), "trend_ma_50"]
        ),
        "bull",
        "bear",
    )

    frame["volatility_flag"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame.loc[
        frame["volatility_median_60d"].notna(),
        "volatility_flag",
    ] = np.where(
        frame.loc[frame["volatility_median_60d"].notna(), "volatility_20d"].ge(
            frame.loc[frame["volatility_median_60d"].notna(), "volatility_median_60d"]
        ),
        "high_vol",
        "low_vol",
    )

    frame["momentum_flag"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame.loc[
        frame["momentum_20d"].notna(),
        "momentum_flag",
    ] = np.where(
        frame.loc[frame["momentum_20d"].notna(), "momentum_20d"].ge(0.0),
        "positive_momentum",
        "negative_momentum",
    )

    frame["drawdown_bucket"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame.loc[
        frame["drawdown_63d"].notna() & frame["drawdown_63d"].ge(-0.02),
        "drawdown_bucket",
    ] = "at_high"
    frame.loc[
        frame["drawdown_63d"].notna()
        & frame["drawdown_63d"].lt(-0.02)
        & frame["drawdown_63d"].gt(-0.10),
        "drawdown_bucket",
    ] = "pullback"
    frame.loc[
        frame["drawdown_63d"].notna() & frame["drawdown_63d"].le(-0.10),
        "drawdown_bucket",
    ] = "deep_drawdown"

    frame["regime"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    regime_mask = frame["trend_flag"].notna() & frame["volatility_flag"].notna()
    frame.loc[regime_mask, "regime"] = (
        frame.loc[regime_mask, "trend_flag"]
        + "_"
        + frame.loc[regime_mask, "volatility_flag"]
    )
    return frame[
        [
            "date",
            "regime",
            "trend_flag",
            "volatility_flag",
            "volatility_20d",
            "benchmark_return",
            "momentum_20d",
            "momentum_flag",
            "drawdown_63d",
            "drawdown_bucket",
        ]
    ]
