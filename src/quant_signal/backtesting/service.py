"""Walk-forward backtesting orchestration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_signal.backtesting.analytics import (
    BACKTEST_ANALYTICS_VERSION,
    BACKTEST_DETAIL_ARTIFACT_COLUMNS,
    BACKTEST_DETAIL_ARTIFACT_VERSION,
    attach_benchmark_relative_analytics,
    build_benchmark_relative_summary,
    build_dimension_summaries,
    build_group_summary,
    build_turnover_daily_metrics,
    build_turnover_summary,
)
from quant_signal.backtesting.execution import BacktestExecutionAssumptions
from quant_signal.backtesting.regimes import (
    REGIME_DEFINITION_VERSION,
    label_regimes,
)
from quant_signal.core.config import Settings, get_settings
from quant_signal.core.hashing import sha256_file, sha256_json
from quant_signal.serving.service import rank_signal_frame
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import BacktestRun
from quant_signal.storage.repositories import BacktestRunRecord, StorageRepository
from quant_signal.training.artifacts import ModelArtifactBundle
from quant_signal.training.service import TrainingService


@dataclass(frozen=True)
class SleevePositionDetail:
    """Raw contribution detail for one symbol inside one sleeve on one active date."""

    signal_date: pd.Timestamp
    active_date: pd.Timestamp
    symbol: str
    rank: int
    gross_return: float
    transaction_cost: float
    slippage_cost: float
    is_entry: bool
    is_exit: bool
    is_held: bool


@dataclass(frozen=True)
class BacktestSimulationResult:
    """Cost-aware portfolio return series plus execution counts."""

    portfolio_returns: pd.DataFrame
    detail_frame: pd.DataFrame
    sleeves_opened: int
    sleeves_closed: int


class BacktestService:
    """Run regime-aware walk-forward backtests for a persisted model version."""

    def __init__(
        self,
        settings: Settings | None = None,
        database_url: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database_url = database_url or self.settings.database_url
        self.training_service = TrainingService(
            settings=self.settings,
            database_url=self.database_url,
        )

    def run(
        self,
        model_version_id: str,
        top_n: int | None = None,
        execution_assumptions: BacktestExecutionAssumptions | None = None,
    ) -> BacktestRun:
        """Run and persist a walk-forward backtest."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            model_version = repository.get_model_version(model_version_id)
            dataset_version = repository.get_dataset_version(model_version.dataset_version_id)

        dataset_frame = pd.read_parquet(dataset_version.artifact_path)
        dataset_frame["date"] = pd.to_datetime(dataset_frame["date"])
        monthly_start_dates = self._monthly_start_dates(dataset_frame["date"])
        top_n_signals = top_n or self.settings.top_n_signals
        resolved_assumptions = (
            execution_assumptions or BacktestExecutionAssumptions.from_settings(self.settings)
        )

        all_ranked_signals: list[pd.DataFrame] = []
        for month_start in monthly_start_dates:
            history = dataset_frame[dataset_frame["date"] < month_start].dropna(
                subset=[model_version.target_column]
            )
            if history["date"].nunique() < self.settings.min_training_days:
                continue

            try:
                split = self.training_service._build_split_for_backtest(
                    history,
                    embargo_days=model_version.horizon_days,
                )
                bundle = self.training_service._fit_model_family(
                    dataset_frame=history,
                    dataset_version_id=dataset_version.id,
                    feature_columns=model_version.feature_columns,
                    target_column=model_version.target_column,
                    horizon=model_version.horizon_days,
                    model_family=model_version.model_family,
                    split=split,
                )
            except ValueError:
                continue

            month_frame = dataset_frame[
                dataset_frame["date"].dt.to_period("M") == month_start.to_period("M")
            ]
            ranked = self._score_month(bundle, month_frame, top_n_signals)
            if not ranked.empty:
                all_ranked_signals.append(ranked)

        ranked_signals = (
            pd.concat(all_ranked_signals, ignore_index=True)
            if all_ranked_signals
            else pd.DataFrame()
        )
        simulation_result = self._simulate_portfolio_returns(
            ranked_signals=ranked_signals,
            market_frame=dataset_frame,
            horizon_days=model_version.horizon_days,
            top_n=top_n_signals,
            execution_assumptions=resolved_assumptions,
        )
        benchmark_analytics = self._build_benchmark_analytics(dataset_frame)
        analytics_frame = attach_benchmark_relative_analytics(
            simulation_result.portfolio_returns,
            benchmark_analytics,
        )
        regime_summary = self._build_regime_summary(analytics_frame)
        summary_json = self._portfolio_summary(analytics_frame)

        artifact_path = self._build_artifact_path(
            model_version_id=model_version_id,
            top_n=top_n_signals,
            execution_assumptions=resolved_assumptions,
        )
        detail_artifact_path = self._build_detail_artifact_path(
            model_version_id=model_version_id,
            top_n=top_n_signals,
            execution_assumptions=resolved_assumptions,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        detail_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        analytics_frame.to_parquet(artifact_path, index=False)
        simulation_result.detail_frame.to_parquet(detail_artifact_path, index=False)

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            return repository.create_backtest_run(
                BacktestRunRecord(
                    model_version_id=model_version_id,
                    horizon_days=model_version.horizon_days,
                    top_n=top_n_signals,
                    min_training_days=self.settings.min_training_days,
                    status="completed",
                    artifact_path=str(artifact_path),
                    artifact_hash=sha256_file(artifact_path),
                    summary_json=summary_json,
                    regime_summary_json=regime_summary,
                    metadata_json={
                        "signal_count": int(len(ranked_signals)),
                        "sleeves_opened": simulation_result.sleeves_opened,
                        "sleeves_closed": simulation_result.sleeves_closed,
                        "benchmark_symbol": self.settings.benchmark_symbol.upper(),
                        "regime_definition_version": REGIME_DEFINITION_VERSION,
                        "backtest_analytics_version": BACKTEST_ANALYTICS_VERSION,
                        "backtest_detail_artifact_version": BACKTEST_DETAIL_ARTIFACT_VERSION,
                        "detail_artifact_path": str(detail_artifact_path),
                        "detail_artifact_hash": sha256_file(detail_artifact_path),
                        "execution_assumptions": resolved_assumptions.to_metadata_json(),
                    },
                )
            )

    def _monthly_start_dates(self, dates: pd.Series) -> list[pd.Timestamp]:
        """Return the first available trading date for each month."""

        month_starts = dates.groupby(dates.dt.to_period("M")).min()
        return list(month_starts.sort_values())

    def _score_month(
        self,
        bundle: ModelArtifactBundle,
        month_frame: pd.DataFrame,
        top_n: int,
    ) -> pd.DataFrame:
        """Score and rank a month of observations for a trained bundle."""

        if month_frame.empty:
            return pd.DataFrame()

        scorable = month_frame.copy()
        coverage = scorable[bundle.feature_columns].notna().sum(axis=1)
        scorable = scorable[coverage >= max(5, len(bundle.feature_columns) // 2)].copy()
        if scorable.empty:
            return pd.DataFrame()

        scorable["score"] = bundle.predict_positive_proba(scorable)
        ranked = rank_signal_frame(scorable[["date", "symbol", "score"]])
        return ranked[ranked["rank"] <= top_n].copy()

    def _simulate_portfolio_returns(
        self,
        ranked_signals: pd.DataFrame,
        market_frame: pd.DataFrame,
        horizon_days: int,
        top_n: int,
        execution_assumptions: BacktestExecutionAssumptions,
    ) -> BacktestSimulationResult:
        """Simulate overlapping long-only sleeves from ranked signals."""

        if ranked_signals.empty:
            return BacktestSimulationResult(
                portfolio_returns=pd.DataFrame(
                    columns=[
                        "date",
                        "gross_return",
                        "transaction_cost",
                        "slippage_cost",
                        "net_return",
                        "active_sleeves",
                        "portfolio_return",
                        "entries_count",
                        "exits_count",
                        "holdings_count",
                        "turnover",
                        "turnover_cost",
                    ]
                ),
                detail_frame=pd.DataFrame(columns=BACKTEST_DETAIL_ARTIFACT_COLUMNS),
                sleeves_opened=0,
                sleeves_closed=0,
            )

        price_frame = market_frame[["date", "symbol", "adjusted_close"]].copy()
        price_frame["date"] = pd.to_datetime(price_frame["date"])
        price_frame = price_frame.sort_values(["symbol", "date"]).reset_index(drop=True)
        price_frame["daily_return"] = price_frame.groupby("symbol")["adjusted_close"].pct_change()

        symbol_history = {
            symbol: group.reset_index(drop=True)
            for symbol, group in price_frame.groupby("symbol", sort=False)
        }
        raw_detail_rows: list[SleevePositionDetail] = []
        sleeves_opened = 0
        sleeves_closed = 0

        for signal_date, signal_rows in ranked_signals.groupby("date"):
            selected_rows = signal_rows.sort_values("rank").head(top_n).copy()
            if selected_rows.empty:
                continue

            selected_symbols = selected_rows["symbol"].tolist()
            rank_map = {
                str(row.symbol): int(row.rank)
                for row in selected_rows.itertuples(index=False)
            }
            for offset in range(1, horizon_days + 1):
                position_details: list[SleevePositionDetail] = []
                for symbol in selected_symbols:
                    history = symbol_history[symbol]
                    current_positions = history.index[history["date"] == signal_date].tolist()
                    if not current_positions:
                        continue
                    next_position = current_positions[0] + offset
                    if next_position >= len(history):
                        continue
                    active_date = history.loc[next_position, "date"]
                    daily_return = history.loc[next_position, "daily_return"]
                    if pd.isna(daily_return):
                        continue
                    position_details.append(
                        SleevePositionDetail(
                            signal_date=pd.Timestamp(signal_date),
                            active_date=pd.Timestamp(active_date),
                            symbol=str(symbol),
                            rank=rank_map[str(symbol)],
                            gross_return=float(daily_return),
                            transaction_cost=0.0,
                            slippage_cost=0.0,
                            is_entry=offset == 1,
                            is_exit=offset == horizon_days,
                            is_held=1 < offset < horizon_days,
                        )
                    )

                if position_details:
                    transaction_cost = (
                        execution_assumptions.transaction_cost_rate
                        if offset in {1, horizon_days}
                        else 0.0
                    )
                    slippage_cost = (
                        execution_assumptions.slippage_rate
                        if offset in {1, horizon_days}
                        else 0.0
                    )
                    if offset == 1:
                        sleeves_opened += 1
                    if offset == horizon_days:
                        sleeves_closed += 1
                    for detail in position_details:
                        raw_detail_rows.append(
                            SleevePositionDetail(
                                signal_date=detail.signal_date,
                                active_date=detail.active_date,
                                symbol=detail.symbol,
                                rank=detail.rank,
                                gross_return=detail.gross_return,
                                transaction_cost=transaction_cost,
                                slippage_cost=slippage_cost,
                                is_entry=detail.is_entry,
                                is_exit=detail.is_exit,
                                is_held=detail.is_held,
                            )
                        )

        detail_frame = self._build_detail_frame(raw_detail_rows)
        if detail_frame.empty:
            portfolio_returns = pd.DataFrame(
                columns=[
                    "date",
                    "gross_return",
                    "transaction_cost",
                    "slippage_cost",
                    "net_return",
                    "active_sleeves",
                    "portfolio_return",
                    "entries_count",
                    "exits_count",
                    "holdings_count",
                    "turnover",
                    "turnover_cost",
                ]
            )
        else:
            portfolio_returns = (
                detail_frame.groupby("active_date", as_index=False)
                .agg(
                    gross_return=("gross_return_contribution", "sum"),
                    transaction_cost=("transaction_cost_contribution", "sum"),
                    slippage_cost=("slippage_cost_contribution", "sum"),
                    net_return=("net_return_contribution", "sum"),
                    active_sleeves=("signal_date", "nunique"),
                )
                .rename(columns={"active_date": "date"})
            )
            portfolio_returns["portfolio_return"] = portfolio_returns["net_return"]
        turnover_metrics = build_turnover_daily_metrics(
            detail_frame,
            execution_assumptions,
        )
        if not turnover_metrics.empty:
            portfolio_returns = portfolio_returns.merge(
                turnover_metrics,
                on="date",
                how="left",
            )
            portfolio_returns[["entries_count", "exits_count", "holdings_count"]] = (
                portfolio_returns[["entries_count", "exits_count", "holdings_count"]]
                .fillna(0)
                .astype(int)
            )
            portfolio_returns[["turnover", "turnover_cost"]] = (
                portfolio_returns[["turnover", "turnover_cost"]].fillna(0.0)
            )
        else:
            for column in (
                "entries_count",
                "exits_count",
                "holdings_count",
                "turnover",
                "turnover_cost",
            ):
                portfolio_returns[column] = 0.0 if "turnover" in column else 0
        portfolio_returns = portfolio_returns.sort_values("date").reset_index(drop=True)

        return BacktestSimulationResult(
            portfolio_returns=portfolio_returns,
            detail_frame=detail_frame,
            sleeves_opened=sleeves_opened,
            sleeves_closed=sleeves_closed,
        )

    def _build_detail_frame(
        self,
        raw_detail_rows: list[SleevePositionDetail],
    ) -> pd.DataFrame:
        """Build a normalized detail artifact from raw sleeve-position rows."""

        if not raw_detail_rows:
            return pd.DataFrame(columns=BACKTEST_DETAIL_ARTIFACT_COLUMNS)

        frame = pd.DataFrame(
            [
                {
                    "signal_date": row.signal_date,
                    "active_date": row.active_date,
                    "symbol": row.symbol,
                    "rank": row.rank,
                    "gross_return": row.gross_return,
                    "transaction_cost": row.transaction_cost,
                    "slippage_cost": row.slippage_cost,
                    "is_entry": row.is_entry,
                    "is_exit": row.is_exit,
                    "is_held": row.is_held,
                }
                for row in raw_detail_rows
            ]
        )
        sleeve_sizes = (
            frame.groupby(["signal_date", "active_date"], as_index=False)
            .size()
            .rename(columns={"size": "sleeve_symbol_count"})
        )
        active_sleeves = (
            frame.groupby("active_date", as_index=False)["signal_date"]
            .nunique()
            .rename(columns={"signal_date": "active_sleeves"})
        )
        frame = frame.merge(sleeve_sizes, on=["signal_date", "active_date"], how="left")
        frame = frame.merge(active_sleeves, on="active_date", how="left")
        sleeve_symbol_count = frame["sleeve_symbol_count"].astype(float)
        active_sleeve_count = frame["active_sleeves"].astype(float)
        frame["weight"] = (
            1.0 / (sleeve_symbol_count * active_sleeve_count)
        )
        frame["gross_return_contribution"] = frame["weight"] * frame["gross_return"]
        frame["transaction_cost_contribution"] = frame["weight"] * frame["transaction_cost"]
        frame["slippage_cost_contribution"] = frame["weight"] * frame["slippage_cost"]
        frame["net_return_contribution"] = (
            frame["gross_return_contribution"]
            - frame["transaction_cost_contribution"]
            - frame["slippage_cost_contribution"]
        )
        return frame[BACKTEST_DETAIL_ARTIFACT_COLUMNS].sort_values(
            ["active_date", "signal_date", "rank", "symbol"]
        ).reset_index(drop=True)

    def _build_artifact_fingerprint(
        self,
        model_version_id: str,
        top_n: int,
        execution_assumptions: BacktestExecutionAssumptions,
    ) -> str:
        """Build a stable artifact fingerprint for one backtest configuration."""

        return sha256_json(
            {
                "model_version_id": model_version_id,
                "top_n": top_n,
                "min_training_days": self.settings.min_training_days,
                "benchmark_symbol": self.settings.benchmark_symbol.upper(),
                "backtest_analytics_version": BACKTEST_ANALYTICS_VERSION,
                "backtest_detail_artifact_version": BACKTEST_DETAIL_ARTIFACT_VERSION,
                "regime_definition_version": REGIME_DEFINITION_VERSION,
                "execution_assumptions": execution_assumptions.to_metadata_json(),
            }
        )[:12]

    def _build_artifact_path(
        self,
        model_version_id: str,
        top_n: int,
        execution_assumptions: BacktestExecutionAssumptions,
    ) -> Path:
        """Build a reproducible artifact path for one backtest configuration."""

        fingerprint = self._build_artifact_fingerprint(
            model_version_id=model_version_id,
            top_n=top_n,
            execution_assumptions=execution_assumptions,
        )
        return (
            Path(self.settings.artifact_root)
            / "backtests"
            / f"backtest_{model_version_id}_{fingerprint}.parquet"
        )

    def _build_detail_artifact_path(
        self,
        model_version_id: str,
        top_n: int,
        execution_assumptions: BacktestExecutionAssumptions,
    ) -> Path:
        """Build a reproducible companion detail artifact path."""

        fingerprint = self._build_artifact_fingerprint(
            model_version_id=model_version_id,
            top_n=top_n,
            execution_assumptions=execution_assumptions,
        )
        return (
            Path(self.settings.artifact_root)
            / "backtests"
            / f"backtest_{model_version_id}_{fingerprint}_detail.parquet"
        )

    def _portfolio_summary(self, portfolio_returns: pd.DataFrame) -> dict[str, object]:
        """Compute summary statistics for a portfolio return series."""

        if portfolio_returns.empty:
            return {
                "cumulative_return": 0.0,
                "gross_cumulative_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": None,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
                "total_transaction_cost": 0.0,
                "total_slippage_cost": 0.0,
                "total_cost_drag": 0.0,
                "benchmark_metrics": build_benchmark_relative_summary(
                    portfolio_returns,
                    self.settings.benchmark_symbol.upper(),
                )["benchmark_metrics"],
                "active_metrics": build_benchmark_relative_summary(
                    portfolio_returns,
                    self.settings.benchmark_symbol.upper(),
                )["active_metrics"],
                "turnover_metrics": build_turnover_summary(portfolio_returns),
                "dimension_summaries": build_dimension_summaries(portfolio_returns),
            }

        gross_returns = portfolio_returns["gross_return"].astype(float)
        returns = portfolio_returns["portfolio_return"].astype(float)
        gross_cumulative_curve = (1.0 + gross_returns).cumprod()
        cumulative_curve = (1.0 + returns).cumprod()
        cumulative_return = float(cumulative_curve.iloc[-1] - 1.0)
        gross_cumulative_return = float(gross_cumulative_curve.iloc[-1] - 1.0)
        annualized_return = float(cumulative_curve.iloc[-1] ** (252 / len(returns)) - 1.0)
        annualized_volatility = float(returns.std(ddof=0) * math.sqrt(252))
        sharpe_ratio = (
            float(annualized_return / annualized_volatility)
            if annualized_volatility > 0
            else None
        )
        drawdown = cumulative_curve / cumulative_curve.cummax() - 1.0
        total_transaction_cost = float(portfolio_returns["transaction_cost"].sum())
        total_slippage_cost = float(portfolio_returns["slippage_cost"].sum())
        benchmark_summary = build_benchmark_relative_summary(
            portfolio_returns,
            self.settings.benchmark_symbol.upper(),
        )
        return {
            "cumulative_return": cumulative_return,
            "gross_cumulative_return": gross_cumulative_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": float(drawdown.min()),
            "hit_rate": float((returns > 0).mean()),
            "total_transaction_cost": total_transaction_cost,
            "total_slippage_cost": total_slippage_cost,
            "total_cost_drag": total_transaction_cost + total_slippage_cost,
            "benchmark_metrics": benchmark_summary["benchmark_metrics"],
            "active_metrics": benchmark_summary["active_metrics"],
            "turnover_metrics": build_turnover_summary(portfolio_returns),
            "dimension_summaries": build_dimension_summaries(portfolio_returns),
        }

    def _build_benchmark_analytics(
        self,
        market_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return benchmark analytics and regime labels for the benchmark symbol."""

        benchmark_symbol = self.settings.benchmark_symbol.upper()
        benchmark_frame = market_frame[
            market_frame["symbol"] == benchmark_symbol
        ].copy()
        if benchmark_frame.empty:
            if market_frame.empty:
                raise ValueError(
                    f"Missing benchmark bars for symbol {benchmark_symbol}."
                )

            start_date = pd.to_datetime(market_frame["date"]).min().date()
            end_date = pd.to_datetime(market_frame["date"]).max().date()
            with session_scope(self.database_url) as session:
                repository = StorageRepository(session)
                benchmark_frame = repository.load_daily_bars_frame(
                    [benchmark_symbol],
                    start_date=start_date,
                    end_date=end_date,
                )

        if benchmark_frame.empty:
            raise ValueError(f"Missing benchmark bars for symbol {benchmark_symbol}.")

        benchmark_frame["date"] = pd.to_datetime(benchmark_frame["date"])
        return label_regimes(benchmark_frame)

    def _build_regime_summary(
        self,
        portfolio_returns: pd.DataFrame,
    ) -> dict[str, dict[str, object]]:
        """Slice portfolio returns by the primary benchmark regime."""

        return build_group_summary(portfolio_returns, "regime")
