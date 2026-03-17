"""Configuration tests."""

from quant_signal.core.config import Settings


def test_settings_defaults() -> None:
    """Default settings should reflect the planned MVP."""

    settings = Settings()

    assert settings.app_name == "quant-signal-workstation"
    assert settings.default_horizons == [1, 5, 20]
    assert settings.benchmark_symbol == "SPY"
    assert settings.market_data_provider == "yfinance"
    assert settings.market_data_max_attempts == 1
    assert settings.market_data_backoff_seconds == 1.0
    assert settings.market_data_backoff_multiplier == 2.0
    assert settings.backtest_transaction_cost_bps == 0.0
    assert settings.backtest_slippage_bps == 0.0


def test_settings_parse_list_env_values(monkeypatch: object) -> None:
    """Comma-delimited environment values should parse into lists."""

    monkeypatch.setenv("UNIVERSE_SYMBOLS", "AAPL, MSFT, spy")
    monkeypatch.setenv("DEFAULT_HORIZONS", "1, 10,20")
    monkeypatch.setenv("MARKET_DATA_PROVIDER", " YFINANCE ")
    monkeypatch.setenv("MARKET_DATA_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("MARKET_DATA_BACKOFF_SECONDS", "1.5")
    monkeypatch.setenv("MARKET_DATA_BACKOFF_MULTIPLIER", "3.0")
    monkeypatch.setenv("BACKTEST_TRANSACTION_COST_BPS", "5.5")
    monkeypatch.setenv("BACKTEST_SLIPPAGE_BPS", "1.25")

    settings = Settings()

    assert settings.universe_symbols == ["AAPL", "MSFT", "SPY"]
    assert settings.default_horizons == [1, 10, 20]
    assert settings.market_data_provider == "yfinance"
    assert settings.market_data_max_attempts == 3
    assert settings.market_data_backoff_seconds == 1.5
    assert settings.market_data_backoff_multiplier == 3.0
    assert settings.backtest_transaction_cost_bps == 5.5
    assert settings.backtest_slippage_bps == 1.25
