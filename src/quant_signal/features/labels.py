"""Target generation for multi-horizon binary forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def target_columns(horizons: Sequence[int]) -> list[str]:
    """Return ordered binary target column names for the given horizons."""

    return [f"target_up_{horizon}d" for horizon in horizons]


def forward_return_columns(horizons: Sequence[int]) -> list[str]:
    """Return ordered forward return column names for the given horizons."""

    return [f"forward_return_{horizon}d" for horizon in horizons]


def add_forward_return_targets(frame: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """Add forward returns and binary up/down targets to a symbol-date frame."""

    labeled = frame.copy()
    grouped = labeled.groupby("symbol", sort=False)

    for horizon in horizons:
        forward_prices = grouped["adjusted_close"].shift(-horizon)
        forward_returns = (forward_prices / labeled["adjusted_close"]) - 1.0
        labeled[f"forward_return_{horizon}d"] = forward_returns
        target = forward_returns.gt(0).astype("float64")
        labeled[f"target_up_{horizon}d"] = target.where(forward_returns.notna())

    return labeled
