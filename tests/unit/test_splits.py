"""Tests for temporal split utilities."""

import pandas as pd

from quant_signal.features.splits import build_temporal_split


def test_build_temporal_split_applies_embargo() -> None:
    """Embargoed temporal splits should leave gaps between partitions."""

    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    frame = pd.DataFrame({"date": dates})

    split = build_temporal_split(
        frame,
        validation_fraction=0.2,
        test_fraction=0.2,
        embargo_days=2,
        minimum_train_dates=20,
    )

    train_end_index = dates.get_loc(split.train_dates[-1])
    validation_start_index = dates.get_loc(split.validation_dates[0])
    validation_end_index = dates.get_loc(split.validation_dates[-1])
    test_start_index = dates.get_loc(split.test_dates[0])

    assert validation_start_index - train_end_index - 1 == 2
    assert test_start_index - validation_end_index - 1 == 2
