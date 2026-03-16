"""Configuration tests."""

from quant_signal.core.config import Settings


def test_settings_defaults() -> None:
    """Default settings should reflect the planned MVP."""

    settings = Settings()

    assert settings.app_name == "quant-signal-workstation"
    assert settings.default_horizons == [1, 5, 20]
    assert settings.benchmark_symbol == "SPY"
