"""Configuration tests."""

from quant_signal.core.config import Settings


def test_settings_defaults() -> None:
    """Default settings should reflect the planned MVP."""

    settings = Settings()

    assert settings.app_name == "quant-signal-workstation"
    assert settings.default_horizons == [1, 5, 20]
    assert settings.benchmark_symbol == "SPY"


def test_settings_parse_list_env_values(monkeypatch: object) -> None:
    """Comma-delimited environment values should parse into lists."""

    monkeypatch.setenv("UNIVERSE_SYMBOLS", "AAPL, MSFT, spy")
    monkeypatch.setenv("DEFAULT_HORIZONS", "1, 10,20")

    settings = Settings()

    assert settings.universe_symbols == ["AAPL", "MSFT", "SPY"]
    assert settings.default_horizons == [1, 10, 20]
