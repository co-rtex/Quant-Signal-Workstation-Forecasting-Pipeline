"""Execution assumptions and helpers for cost-aware backtests."""

from __future__ import annotations

from dataclasses import dataclass

from quant_signal.core.config import Settings


@dataclass(frozen=True)
class BacktestExecutionAssumptions:
    """Explicit execution cost assumptions for a backtest run."""

    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0

    @classmethod
    def from_settings(cls, settings: Settings) -> BacktestExecutionAssumptions:
        """Build execution assumptions from application settings."""

        return cls(
            transaction_cost_bps=float(settings.backtest_transaction_cost_bps),
            slippage_bps=float(settings.backtest_slippage_bps),
        )

    @staticmethod
    def bps_to_rate(value: float) -> float:
        """Convert basis points into a decimal rate."""

        return float(value) / 10_000.0

    @property
    def transaction_cost_rate(self) -> float:
        """Return the transaction cost rate for one portfolio side."""

        return self.bps_to_rate(self.transaction_cost_bps)

    @property
    def slippage_rate(self) -> float:
        """Return the slippage rate for one portfolio side."""

        return self.bps_to_rate(self.slippage_bps)

    @property
    def total_cost_rate_per_side(self) -> float:
        """Return the combined transaction and slippage cost rate per side."""

        return self.transaction_cost_rate + self.slippage_rate

    def to_metadata_json(self) -> dict[str, float]:
        """Return a JSON-serializable snapshot of resolved assumptions."""

        return {
            "transaction_cost_bps": float(self.transaction_cost_bps),
            "slippage_bps": float(self.slippage_bps),
            "transaction_cost_rate": self.transaction_cost_rate,
            "slippage_rate": self.slippage_rate,
            "total_cost_rate_per_side": self.total_cost_rate_per_side,
        }
