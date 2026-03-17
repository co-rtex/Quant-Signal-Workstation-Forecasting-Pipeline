"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        enable_decoding=False,
    )

    app_env: str = Field(default="local")
    app_name: str = Field(default="quant-signal-workstation")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/quant_signal"
    )
    artifact_root: Path = Field(default=Path("artifacts"))
    benchmark_symbol: str = Field(default="SPY")
    universe_symbols: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "SPY"]
    )
    default_horizons: list[int] = Field(default_factory=lambda: [1, 5, 20])
    top_n_signals: int = Field(default=10)
    min_training_days: int = Field(default=252)
    backtest_transaction_cost_bps: float = Field(default=0.0)
    backtest_slippage_bps: float = Field(default=0.0)

    @field_validator("universe_symbols", mode="before")
    @classmethod
    def parse_symbol_list(cls, value: object) -> object:
        """Parse comma-delimited symbol lists from environment variables."""

        if isinstance(value, str):
            return [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]
        return value

    @field_validator("default_horizons", mode="before")
    @classmethod
    def parse_horizon_list(cls, value: object) -> object:
        """Parse comma-delimited horizon lists from environment variables."""

        if isinstance(value, str):
            return [int(horizon.strip()) for horizon in value.split(",") if horizon.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""

    return Settings()
