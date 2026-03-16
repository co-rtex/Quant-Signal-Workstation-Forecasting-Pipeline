"""Temporal split utilities for leakage-safe model validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Discrete train, validation, and test date partitions."""

    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    embargo_days: int

    def mask(self, frame: pd.DataFrame, split_name: str) -> pd.Series:
        """Return a boolean mask for the requested split name."""

        dates = set(getattr(self, f"{split_name}_dates"))
        return pd.to_datetime(frame["date"]).isin(dates)


def build_temporal_split(
    frame: pd.DataFrame,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    embargo_days: int = 20,
    minimum_train_dates: int = 60,
) -> TemporalSplit:
    """Construct embargoed train, validation, and test date partitions."""

    unique_dates = tuple(sorted(pd.to_datetime(frame["date"]).drop_duplicates()))
    total_dates = len(unique_dates)
    validation_size = max(1, ceil(total_dates * validation_fraction))
    test_size = max(1, ceil(total_dates * test_fraction))
    train_size = total_dates - validation_size - test_size - (2 * embargo_days)

    if train_size < minimum_train_dates:
        raise ValueError(
            "Not enough unique dates to build a temporal split with the requested embargo."
        )

    validation_start = train_size + embargo_days
    validation_end = validation_start + validation_size
    test_start = validation_end + embargo_days
    test_end = test_start + test_size

    return TemporalSplit(
        train_dates=unique_dates[:train_size],
        validation_dates=unique_dates[validation_start:validation_end],
        test_dates=unique_dates[test_start:test_end],
        embargo_days=embargo_days,
    )
