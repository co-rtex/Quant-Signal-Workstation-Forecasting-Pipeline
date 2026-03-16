"""Market data provider contracts and development adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from quant_signal.ingestion.models import MarketDataBar


class MarketDataProvider(ABC):
    """Abstract market data provider."""

    name: str

    @abstractmethod
    def fetch_daily_bars(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> list[MarketDataBar]:
        """Fetch normalized daily bars for the requested symbols."""


class YFinanceMarketDataProvider(MarketDataProvider):
    """Free development adapter backed by yfinance."""

    name = "yfinance"

    def fetch_daily_bars(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> list[MarketDataBar]:
        """Fetch daily OHLCV bars from Yahoo Finance."""

        requested = sorted({symbol.upper() for symbol in symbols})
        inclusive_end = end_date + timedelta(days=1)
        bars: list[MarketDataBar] = []

        for symbol in requested:
            frame = yf.download(
                symbol,
                start=start_date.isoformat(),
                end=inclusive_end.isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            if frame.empty:
                continue

            normalized = frame.reset_index().rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adjusted_close",
                    "Volume": "volume",
                }
            )
            normalized["date"] = pd.to_datetime(normalized["date"]).dt.date

            for record in normalized.to_dict(orient="records"):
                bars.append(
                    MarketDataBar(
                        symbol=symbol,
                        trade_date=record["date"],
                        open=float(record["open"]),
                        high=float(record["high"]),
                        low=float(record["low"]),
                        close=float(record["close"]),
                        adjusted_close=float(record.get("adjusted_close", record["close"])),
                        volume=int(record["volume"]),
                    )
                )

        return bars
