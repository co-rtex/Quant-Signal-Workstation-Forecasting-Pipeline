"""Market regime labeling utilities."""

from __future__ import annotations

import math

import pandas as pd


def label_regimes(benchmark_frame: pd.DataFrame) -> pd.DataFrame:
    """Label each benchmark date with a simple trend/volatility regime."""

    frame = benchmark_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["benchmark_daily_return"] = frame["adjusted_close"].pct_change()
    frame["trend_ma_50"] = frame["adjusted_close"].rolling(50, min_periods=50).mean()
    frame["volatility_20d"] = (
        frame["benchmark_daily_return"].rolling(20, min_periods=20).std() * math.sqrt(252)
    )
    frame["volatility_median_60d"] = frame["volatility_20d"].rolling(60, min_periods=20).median()
    frame["trend_flag"] = frame["adjusted_close"].ge(frame["trend_ma_50"]).map(
        {True: "bull", False: "bear"}
    )
    frame["volatility_flag"] = frame["volatility_20d"].ge(frame["volatility_median_60d"]).map(
        {True: "high_vol", False: "low_vol"}
    )
    frame["regime"] = frame["trend_flag"] + "_" + frame["volatility_flag"]
    return frame[["date", "regime", "trend_flag", "volatility_flag", "volatility_20d"]]
