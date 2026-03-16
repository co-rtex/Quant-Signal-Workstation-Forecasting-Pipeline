"""Tests for target generation."""

from datetime import date

import pandas as pd

from quant_signal.features.labels import add_forward_return_targets


def test_add_forward_return_targets() -> None:
    """Forward returns and binary targets should align to future prices."""

    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "date": [date(2024, 1, day) for day in range(2, 6)],
            "adjusted_close": [100.0, 110.0, 99.0, 120.0],
        }
    )

    labeled = add_forward_return_targets(frame, [1, 2])

    assert round(float(labeled.loc[0, "forward_return_1d"]), 6) == 0.1
    assert round(float(labeled.loc[1, "forward_return_2d"]), 6) == round(120.0 / 110.0 - 1.0, 6)
    assert labeled.loc[0, "target_up_1d"] == 1.0
    assert pd.isna(labeled.loc[3, "target_up_1d"])
