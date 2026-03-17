"""Configuration tests."""

from quant_signal.core.config import Settings


def test_settings_defaults() -> None:
    """Default settings should reflect the planned MVP."""

    settings = Settings()

    assert settings.app_name == "quant-signal-workstation"
    assert settings.default_horizons == [1, 5, 20]
    assert settings.benchmark_symbol == "SPY"
    assert settings.backtest_transaction_cost_bps == 0.0
    assert settings.backtest_slippage_bps == 0.0


def test_settings_parse_list_env_values(monkeypatch: object) -> None:
    """Comma-delimited environment values should parse into lists."""

    monkeypatch.setenv("UNIVERSE_SYMBOLS", "AAPL, MSFT, spy")
    monkeypatch.setenv("DEFAULT_HORIZONS", "1, 10,20")
    monkeypatch.setenv("BACKTEST_TRANSACTION_COST_BPS", "5.5")
    monkeypatch.setenv("BACKTEST_SLIPPAGE_BPS", "1.25")

    settings = Settings()

    assert settings.universe_symbols == ["AAPL", "MSFT", "SPY"]
    assert settings.default_horizons == [1, 10, 20]
    assert settings.backtest_transaction_cost_bps == 5.5
    assert settings.backtest_slippage_bps == 1.25
