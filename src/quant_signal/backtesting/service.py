"""Walk-forward backtesting orchestration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_signal.backtesting.regimes import label_regimes
from quant_signal.core.config import Settings, get_settings
from quant_signal.core.hashing import sha256_file
from quant_signal.serving.service import rank_signal_frame
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import BacktestRun
from quant_signal.storage.repositories import BacktestRunRecord, StorageRepository
from quant_signal.training.artifacts import ModelArtifactBundle
from quant_signal.training.service import TrainingService


@dataclass
class SleeveReturn:
    """Daily return for one overlapping holding sleeve."""

    active_date: pd.Timestamp
    sleeve_return: float


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

    def run(self, model_version_id: str, top_n: int | None = None) -> BacktestRun:
        """Run and persist a walk-forward backtest."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            model_version = repository.get_model_version(model_version_id)
            dataset_version = repository.get_dataset_version(model_version.dataset_version_id)

        dataset_frame = pd.read_parquet(dataset_version.artifact_path)
        dataset_frame["date"] = pd.to_datetime(dataset_frame["date"])
        monthly_start_dates = self._monthly_start_dates(dataset_frame["date"])
        top_n_signals = top_n or self.settings.top_n_signals

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
        portfolio_returns = self._simulate_portfolio_returns(
            ranked_signals=ranked_signals,
            market_frame=dataset_frame,
            horizon_days=model_version.horizon_days,
            top_n=top_n_signals,
        )
        regime_summary = self._build_regime_summary(dataset_frame, portfolio_returns)
        summary_json = self._portfolio_summary(portfolio_returns)

        artifact_path = (
            Path(self.settings.artifact_root) / "backtests" / f"backtest_{model_version_id}.parquet"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        portfolio_returns.to_parquet(artifact_path, index=False)

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
                    metadata_json={"signal_count": int(len(ranked_signals))},
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
    ) -> pd.DataFrame:
        """Simulate overlapping long-only sleeves from ranked signals."""

        if ranked_signals.empty:
            return pd.DataFrame(columns=["date", "portfolio_return"])

        price_frame = market_frame[["date", "symbol", "adjusted_close"]].copy()
        price_frame["date"] = pd.to_datetime(price_frame["date"])
        price_frame = price_frame.sort_values(["symbol", "date"]).reset_index(drop=True)
        price_frame["daily_return"] = price_frame.groupby("symbol")["adjusted_close"].pct_change()

        symbol_history = {
            symbol: group.reset_index(drop=True)
            for symbol, group in price_frame.groupby("symbol", sort=False)
        }
        sleeve_returns_by_date: dict[pd.Timestamp, list[float]] = defaultdict(list)

        for signal_date, signal_rows in ranked_signals.groupby("date"):
            selected_symbols = signal_rows.sort_values("rank").head(top_n)["symbol"].tolist()
            if not selected_symbols:
                continue

            sleeve_series: list[SleeveReturn] = []
            for offset in range(1, horizon_days + 1):
                symbol_returns: list[float] = []
                active_date: pd.Timestamp | None = None
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
                    if pd.notna(daily_return):
                        symbol_returns.append(float(daily_return))

                if active_date is not None and symbol_returns:
                    sleeve_series.append(
                        SleeveReturn(
                            active_date=active_date,
                            sleeve_return=float(np.mean(symbol_returns)),
                        )
                    )

            for sleeve_return in sleeve_series:
                sleeve_returns_by_date[sleeve_return.active_date].append(sleeve_return.sleeve_return)

        rows = [
            {"date": active_date, "portfolio_return": float(np.mean(sleeve_returns))}
            for active_date, sleeve_returns in sorted(sleeve_returns_by_date.items())
            if sleeve_returns
        ]
        return pd.DataFrame(rows)

    def _portfolio_summary(self, portfolio_returns: pd.DataFrame) -> dict[str, object]:
        """Compute summary statistics for a portfolio return series."""

        if portfolio_returns.empty:
            return {
                "cumulative_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": None,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
            }

        returns = portfolio_returns["portfolio_return"].astype(float)
        cumulative_curve = (1.0 + returns).cumprod()
        cumulative_return = float(cumulative_curve.iloc[-1] - 1.0)
        annualized_return = float(cumulative_curve.iloc[-1] ** (252 / len(returns)) - 1.0)
        annualized_volatility = float(returns.std(ddof=0) * np.sqrt(252))
        sharpe_ratio = (
            float(annualized_return / annualized_volatility)
            if annualized_volatility > 0
            else None
        )
        drawdown = cumulative_curve / cumulative_curve.cummax() - 1.0
        return {
            "cumulative_return": cumulative_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": float(drawdown.min()),
            "hit_rate": float((returns > 0).mean()),
        }

    def _build_regime_summary(
        self,
        market_frame: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
    ) -> dict[str, object]:
        """Slice portfolio returns by benchmark regime."""

        if portfolio_returns.empty:
            return {}

        benchmark = market_frame[
            market_frame["symbol"] == self.settings.benchmark_symbol.upper()
        ].copy()
        regimes = label_regimes(benchmark)
        merged = portfolio_returns.merge(regimes, on="date", how="left")

        summary: dict[str, object] = {}
        for regime, frame in merged.dropna(subset=["regime"]).groupby("regime"):
            summary[str(regime)] = {
                "sample_count": int(len(frame)),
                "average_return": float(frame["portfolio_return"].mean()),
                "hit_rate": float((frame["portfolio_return"] > 0).mean()),
            }
        return summary
