"""Data structures for market data ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class MarketDataBar:
    """Provider-agnostic normalized daily OHLCV bar."""

    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    source_updated_at: datetime | None = None
