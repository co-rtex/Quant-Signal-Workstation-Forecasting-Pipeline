"""Tests for regime labeling."""

import pandas as pd

from quant_signal.backtesting.regimes import label_regimes


def test_label_regimes_returns_regime_column() -> None:
    """Regime labeling should emit non-empty regime metadata."""

    dates = pd.bdate_range("2024-01-01", periods=80)
    frame = pd.DataFrame(
        {
            "date": dates,
            "adjusted_close": [
                100 + (index * 0.4) + ((index % 6) * 0.2)
                for index in range(len(dates))
            ],
        }
    )

    labeled = label_regimes(frame)

    assert "regime" in labeled.columns
    assert labeled["regime"].notna().sum() > 0
