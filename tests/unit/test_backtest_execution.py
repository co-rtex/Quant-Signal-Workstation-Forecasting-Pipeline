"""Tests for backtest execution assumptions."""

from quant_signal.backtesting.execution import BacktestExecutionAssumptions
from quant_signal.core.config import Settings


def test_execution_assumptions_default_to_zero_costs() -> None:
    """Default execution assumptions should preserve MVP behavior."""

    assumptions = BacktestExecutionAssumptions()

    assert assumptions.transaction_cost_rate == 0.0
    assert assumptions.slippage_rate == 0.0
    assert assumptions.total_cost_rate_per_side == 0.0


def test_execution_assumptions_convert_basis_points_to_rates() -> None:
    """Basis-point assumptions should convert into decimal return deductions."""

    assumptions = BacktestExecutionAssumptions(
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
    )

    assert assumptions.transaction_cost_rate == 0.0005
    assert assumptions.slippage_rate == 0.0002
    assert assumptions.total_cost_rate_per_side == 0.0007


def test_execution_assumptions_load_from_settings() -> None:
    """Settings-backed defaults should hydrate the execution assumptions object."""

    settings = Settings(
        backtest_transaction_cost_bps=4.0,
        backtest_slippage_bps=1.5,
    )

    assumptions = BacktestExecutionAssumptions.from_settings(settings)

    assert assumptions.to_metadata_json() == {
        "transaction_cost_bps": 4.0,
        "slippage_bps": 1.5,
        "transaction_cost_rate": 0.0004,
        "slippage_rate": 0.00015,
        "total_cost_rate_per_side": 0.00055,
    }
