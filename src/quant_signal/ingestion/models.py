"""Data structures for market data ingestion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ProviderFetchResult:
    """Normalized provider fetch output with metadata for persistence."""

    bars: list[MarketDataBar]
    returned_symbols: list[str]
    provider_metadata: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_bars(
        cls,
        bars: Sequence[MarketDataBar],
        provider_metadata: dict[str, object] | None = None,
        warnings: Sequence[str] | None = None,
    ) -> ProviderFetchResult:
        """Build a fetch result from a sequence of normalized bars."""

        normalized_bars = list(bars)
        returned_symbols = sorted({bar.symbol.upper() for bar in normalized_bars})
        return cls(
            bars=normalized_bars,
            returned_symbols=returned_symbols,
            provider_metadata=provider_metadata or {},
            warnings=list(warnings or []),
        )
