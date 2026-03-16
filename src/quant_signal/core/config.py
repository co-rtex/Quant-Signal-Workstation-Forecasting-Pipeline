"""Application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_env: str = Field(default="local")
    app_name: str = Field(default="quant-signal-workstation")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/quant_signal"
    )
    artifact_root: str = Field(default="artifacts")
    benchmark_symbol: str = Field(default="SPY")
    universe_symbols: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "SPY"]
    )
    default_horizons: list[int] = Field(default_factory=lambda: [1, 5, 20])
    top_n_signals: int = Field(default=10)
    min_training_days: int = Field(default=252)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""

    return Settings()
