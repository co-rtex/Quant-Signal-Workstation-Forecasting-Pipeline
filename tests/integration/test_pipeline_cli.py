"""Integration tests for the pipeline CLI."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from quant_signal.cli.pipeline import ServiceFactories, main
from quant_signal.core.config import Settings
from quant_signal.features.pipeline import FeaturePipeline
from quant_signal.ingestion.errors import ProviderPermanentError
from quant_signal.ingestion.models import MarketDataBar, ProviderFetchResult
from quant_signal.ingestion.providers import MarketDataProvider
from quant_signal.ingestion.service import IngestionService
from quant_signal.storage.db import create_all_tables, session_scope
from quant_signal.storage.models import (
    BacktestRun,
    DatasetVersion,
    IngestionRun,
    ModelVersion,
    ShapRun,
    SignalSnapshot,
)
from quant_signal.training.service import TrainingService


class StaticProvider(MarketDataProvider):
    """Deterministic provider for CLI integration tests."""

    name = "static"

    def __init__(self, bars: Sequence[MarketDataBar]) -> None:
        self._bars = list(bars)

    def fetch_daily_bars(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> ProviderFetchResult:
        requested = {symbol.upper() for symbol in symbols}
        filtered_bars = [
            bar
            for bar in self._bars
            if bar.symbol.upper() in requested and start_date <= bar.trade_date <= end_date
        ]
        return ProviderFetchResult.from_bars(
            filtered_bars,
            provider_metadata={"fixture": "synthetic"},
        )


class PermanentFailingProvider(MarketDataProvider):
    """Provider that fails with a terminal provider error."""

    name = "permanent-static"

    def fetch_daily_bars(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> ProviderFetchResult:
        raise ProviderPermanentError(
            self.name,
            "invalid provider response payload",
            cause_type="ValueError",
        )


def build_synthetic_bars() -> list[MarketDataBar]:
    """Build a deterministic synthetic market history."""

    bars: list[MarketDataBar] = []
    dates = pd.bdate_range("2024-01-02", periods=90)

    for index, timestamp in enumerate(dates):
        trade_date = timestamp.date()
        aapl_close = 100.0 + (index * 0.7) + ((index % 5) * 0.25)
        spy_close = 400.0 + (index * 0.3) + ((index % 7) * 0.15)

        bars.append(
            MarketDataBar(
                symbol="AAPL",
                trade_date=trade_date,
                open=aapl_close - 0.6,
                high=aapl_close + 0.8,
                low=aapl_close - 1.1,
                close=aapl_close,
                adjusted_close=aapl_close,
                volume=1_000_000 + (index * 1_000),
            )
        )
        bars.append(
            MarketDataBar(
                symbol="SPY",
                trade_date=trade_date,
                open=spy_close - 0.8,
                high=spy_close + 1.0,
                low=spy_close - 1.2,
                close=spy_close,
                adjusted_close=spy_close,
                volume=5_000_000 + (index * 2_000),
            )
        )

    return bars


def build_training_bars(periods: int = 220) -> list[MarketDataBar]:
    """Build a longer deterministic history with enough variation for training."""

    bars: list[MarketDataBar] = []
    dates = pd.bdate_range("2023-01-03", periods=periods)

    for index, timestamp in enumerate(dates):
        trade_date = timestamp.date()
        aapl_close = 95.0 + (index * 0.18) + (math.sin(index / 5.0) * 7.5)
        spy_close = 390.0 + (index * 0.1) + (math.sin(index / 8.0) * 4.0)

        bars.append(
            MarketDataBar(
                symbol="AAPL",
                trade_date=trade_date,
                open=aapl_close - 0.5,
                high=aapl_close + 0.9,
                low=aapl_close - 1.1,
                close=aapl_close,
                adjusted_close=aapl_close,
                volume=1_500_000 + (index * 500),
            )
        )
        bars.append(
            MarketDataBar(
                symbol="SPY",
                trade_date=trade_date,
                open=spy_close - 0.6,
                high=spy_close + 0.8,
                low=spy_close - 1.0,
                close=spy_close,
                adjusted_close=spy_close,
                volume=6_000_000 + (index * 1_000),
            )
        )

    return bars


def ingest_synthetic_history(settings: Settings) -> None:
    """Persist deterministic bars for CLI tests that need upstream market history."""

    IngestionService(
        provider=StaticProvider(build_synthetic_bars()),
        settings=settings,
        sleep_fn=lambda _: None,
    ).ingest_daily_bars(
        ["AAPL"],
        date(2024, 1, 2),
        date(2024, 5, 31),
    )


def ingest_training_history(settings: Settings) -> None:
    """Persist longer deterministic bars for training-oriented CLI tests."""

    IngestionService(
        provider=StaticProvider(build_training_bars()),
        settings=settings,
        sleep_fn=lambda _: None,
    ).ingest_daily_bars(
        ["AAPL"],
        date(2023, 1, 3),
        date(2023, 11, 30),
    )


def test_pipeline_ingest_command_persists_run_and_emits_json_summary(tmp_path: Path) -> None:
    """The ingest command should print a machine-readable success summary."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
    )
    create_all_tables(database_url)

    stdout = StringIO()
    exit_code = main(
        [
            "ingest",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-05-31",
            "--symbols",
            "AAPL",
        ],
        settings=settings,
        stdout=stdout,
        service_factories=ServiceFactories(
            ingestion=lambda resolved_settings: IngestionService(
                provider=StaticProvider(build_synthetic_bars()),
                settings=resolved_settings,
                sleep_fn=lambda _: None,
            )
        ),
    )

    assert exit_code == 0

    payload = json.loads(stdout.getvalue())
    assert payload == {
        "benchmark_symbol": "SPY",
        "command": "ingest",
        "completed_after_retry": False,
        "end_date": "2024-05-31",
        "fetch_symbols": ["AAPL", "SPY"],
        "provider": "static",
        "records_written": 180,
        "requested_symbols": ["AAPL"],
        "retry_attempt_count": 1,
        "run_id": payload["run_id"],
        "start_date": "2024-01-02",
        "status": "completed",
    }

    with session_scope(database_url) as session:
        run = session.scalar(select(IngestionRun).where(IngestionRun.id == payload["run_id"]))

    assert run is not None
    assert run.status == "completed"
    assert run.records_written == 180


def test_pipeline_ingest_command_defaults_to_settings_universe(tmp_path: Path) -> None:
    """The ingest command should use configured universe symbols when none are provided."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-default.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
    )
    create_all_tables(database_url)

    stdout = StringIO()
    exit_code = main(
        [
            "ingest",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-05-31",
        ],
        settings=settings,
        stdout=stdout,
        service_factories=ServiceFactories(
            ingestion=lambda resolved_settings: IngestionService(
                provider=StaticProvider(build_synthetic_bars()),
                settings=resolved_settings,
                sleep_fn=lambda _: None,
            )
        ),
    )

    assert exit_code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["requested_symbols"] == ["AAPL", "SPY"]
    assert payload["fetch_symbols"] == ["AAPL", "SPY"]
    assert payload["records_written"] == 180


def test_pipeline_ingest_command_returns_nonzero_and_logs_failures(
    tmp_path: Path,
) -> None:
    """The ingest command should return a non-zero exit code on service failures."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-failed.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
    )
    create_all_tables(database_url)

    stdout = StringIO()
    stderr = StringIO()
    exit_code = main(
        [
            "ingest",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-05-31",
            "--symbols",
            "AAPL",
        ],
        settings=settings,
        stdout=stdout,
        stderr=stderr,
        service_factories=ServiceFactories(
            ingestion=lambda resolved_settings: IngestionService(
                provider=PermanentFailingProvider(),
                settings=resolved_settings,
                sleep_fn=lambda _: None,
            )
        ),
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert json.loads(stderr.getvalue()) == {
        "command": "ingest",
        "error": "invalid provider response payload",
        "error_type": "ProviderPermanentError",
        "status": "failed",
    }

    with session_scope(database_url) as session:
        run = session.scalar(select(IngestionRun).order_by(IngestionRun.started_at.desc()))

    assert run is not None
    assert run.status == "failed"
    assert run.metadata_json["retry"]["attempt_count"] == 1
    assert run.metadata_json["failure"] == {
        "error": "invalid provider response payload",
        "error_type": "ValueError",
        "retriable": False,
    }


def test_pipeline_build_dataset_command_persists_manifest_and_emits_json_summary(
    tmp_path: Path,
) -> None:
    """The dataset command should print a machine-readable dataset summary."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-dataset.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
        default_horizons=[1, 5, 20],
    )
    create_all_tables(database_url)
    ingest_synthetic_history(settings)

    stdout = StringIO()
    exit_code = main(
        [
            "build-dataset",
            "--as-of-date",
            "2024-05-31",
            "--symbols",
            "AAPL",
            "--feature-set-version",
            "cli_test_v1",
        ],
        settings=settings,
        stdout=stdout,
    )

    assert exit_code == 0

    payload = json.loads(stdout.getvalue())
    assert payload["command"] == "build-dataset"
    assert payload["status"] == "completed"
    assert payload["as_of_date"] == "2024-05-31"
    assert payload["feature_set_version"] == "cli_test_v1"
    assert payload["symbols"] == ["AAPL"]
    assert payload["horizons"] == [1, 5, 20]
    assert payload["benchmark_symbol"] == "SPY"
    assert payload["row_count"] > 0
    assert Path(payload["artifact_path"]).exists()

    with session_scope(database_url) as session:
        dataset = session.scalar(
            select(DatasetVersion).where(DatasetVersion.id == payload["dataset_version_id"])
        )

    assert dataset is not None
    assert dataset.feature_set_version == "cli_test_v1"
    assert dataset.row_count == payload["row_count"]
    assert dataset.artifact_path == payload["artifact_path"]
    assert dataset.artifact_hash == payload["artifact_hash"]
    assert dataset.symbols == payload["symbols"]
    assert dataset.horizons == payload["horizons"]
    assert dataset.metadata_json["date_range"] == payload["date_range"]


def test_pipeline_build_dataset_command_returns_nonzero_when_no_bars_exist(
    tmp_path: Path,
) -> None:
    """The dataset command should fail cleanly when no persisted bars are available."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-dataset-empty.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
    )
    create_all_tables(database_url)

    stdout = StringIO()
    stderr = StringIO()
    exit_code = main(
        [
            "build-dataset",
            "--as-of-date",
            "2024-05-31",
            "--symbols",
            "AAPL",
        ],
        settings=settings,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert json.loads(stderr.getvalue()) == {
        "command": "build-dataset",
        "error": "No persisted bars were found for the requested dataset build.",
        "error_type": "ValueError",
        "status": "failed",
    }

    with session_scope(database_url) as session:
        dataset = session.scalar(select(DatasetVersion).order_by(DatasetVersion.created_at.desc()))

    assert dataset is None


def test_pipeline_train_command_persists_models_and_emits_json_summary(tmp_path: Path) -> None:
    """The train command should print a machine-readable model summary."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-train.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
        default_horizons=[1, 5, 20],
        min_training_days=80,
    )
    create_all_tables(database_url)
    ingest_training_history(settings)
    dataset = FeaturePipeline(settings=settings).build_dataset(date(2023, 11, 30), ["AAPL"])

    stdout = StringIO()
    exit_code = main(
        [
            "train",
            "--dataset-version-id",
            dataset.id,
            "--horizon",
            "5",
            "--horizon",
            "5",
            "--horizon",
            "1",
        ],
        settings=settings,
        stdout=stdout,
    )

    assert exit_code == 0

    payload = json.loads(stdout.getvalue())
    assert payload["command"] == "train"
    assert payload["status"] == "completed"
    assert payload["dataset_version_id"] == dataset.id
    assert payload["horizons"] == [1, 5]
    assert payload["model_count"] == 4
    assert len(payload["champion_models"]) == 2
    assert payload["champion_models"][0]["horizon_days"] == 1
    assert payload["champion_models"][1]["horizon_days"] == 5
    assert all(model["model_version_id"] for model in payload["models"])
    assert all(Path(model["artifact_path"]).exists() for model in payload["models"])

    with session_scope(database_url) as session:
        models = list(
            session.execute(
                select(ModelVersion)
                .where(ModelVersion.dataset_version_id == dataset.id)
                .order_by(
                    ModelVersion.horizon_days,
                    ModelVersion.champion_rank,
                    ModelVersion.model_family,
                )
            ).scalars()
        )
        champion_snapshots = list(
            session.execute(
                select(SignalSnapshot)
                .where(SignalSnapshot.model_version_id.in_([model.id for model in models]))
            ).scalars()
        )

    assert len(models) == payload["model_count"]
    assert {model.id for model in models} == {
        model_payload["model_version_id"] for model_payload in payload["models"]
    }
    assert champion_snapshots
    assert {
        snapshot.model_version_id for snapshot in champion_snapshots
    } == {
        model_payload["model_version_id"] for model_payload in payload["champion_models"]
    }


def test_pipeline_train_command_returns_nonzero_for_unknown_dataset_id(
    tmp_path: Path,
) -> None:
    """The train command should fail cleanly for an unknown dataset version."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-train-failed.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
    )
    create_all_tables(database_url)

    stdout = StringIO()
    stderr = StringIO()
    exit_code = main(
        [
            "train",
            "--dataset-version-id",
            "missing-dataset-id",
        ],
        settings=settings,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert json.loads(stderr.getvalue()) == {
        "command": "train",
        "error": "Unknown dataset version: missing-dataset-id",
        "error_type": "ValueError",
        "status": "failed",
    }


def test_pipeline_backtest_command_persists_run_and_emits_json_summary(
    tmp_path: Path,
) -> None:
    """The backtest command should print a machine-readable run summary."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-backtest.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
        default_horizons=[1, 5, 20],
        min_training_days=80,
        top_n_signals=1,
    )
    create_all_tables(database_url)
    ingest_training_history(settings)
    dataset = FeaturePipeline(settings=settings).build_dataset(date(2023, 11, 30), ["AAPL"])
    trained_models = TrainingService(settings=settings).train(dataset.id, [5])
    champion_model = next(model for model in trained_models if model.champion_rank == 1)

    stdout = StringIO()
    exit_code = main(
        [
            "backtest",
            "--model-version-id",
            champion_model.id,
            "--top-n",
            "1",
            "--transaction-cost-bps",
            "5",
            "--slippage-bps",
            "2",
        ],
        settings=settings,
        stdout=stdout,
    )

    assert exit_code == 0

    payload = json.loads(stdout.getvalue())
    assert payload["command"] == "backtest"
    assert payload["status"] == "completed"
    assert payload["model_version_id"] == champion_model.id
    assert payload["horizon_days"] == champion_model.horizon_days
    assert payload["top_n"] == 1
    assert payload["benchmark_symbol"] == "SPY"
    assert payload["signal_count"] >= 0
    assert payload["execution_assumptions"] == {
        "slippage_bps": 2.0,
        "slippage_rate": 0.0002,
        "total_cost_rate_per_side": 0.0007,
        "transaction_cost_bps": 5.0,
        "transaction_cost_rate": 0.0005,
    }
    assert set(payload["summary"]) == {
        "annualized_return",
        "annualized_volatility",
        "cumulative_return",
        "gross_cumulative_return",
        "hit_rate",
        "max_drawdown",
        "sharpe_ratio",
    }
    assert Path(payload["artifact_path"]).exists()
    assert Path(payload["detail_artifact_path"]).exists()

    with session_scope(database_url) as session:
        backtest_run = session.scalar(
            select(BacktestRun).where(BacktestRun.id == payload["backtest_run_id"])
        )

    assert backtest_run is not None
    assert backtest_run.model_version_id == champion_model.id
    assert backtest_run.top_n == payload["top_n"]
    assert backtest_run.artifact_path == payload["artifact_path"]
    assert backtest_run.artifact_hash == payload["artifact_hash"]
    assert backtest_run.metadata_json["detail_artifact_path"] == payload["detail_artifact_path"]
    assert backtest_run.metadata_json["detail_artifact_hash"] == payload["detail_artifact_hash"]
    assert backtest_run.metadata_json["execution_assumptions"] == payload["execution_assumptions"]
    assert backtest_run.summary_json["cumulative_return"] == payload["summary"]["cumulative_return"]


def test_pipeline_backtest_command_returns_nonzero_for_unknown_model_id(
    tmp_path: Path,
) -> None:
    """The backtest command should fail cleanly for an unknown model version."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-backtest-failed.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
    )
    create_all_tables(database_url)

    stdout = StringIO()
    stderr = StringIO()
    exit_code = main(
        [
            "backtest",
            "--model-version-id",
            "missing-model-version",
        ],
        settings=settings,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert json.loads(stderr.getvalue()) == {
        "command": "backtest",
        "error": "Unknown model version: missing-model-version",
        "error_type": "ValueError",
        "status": "failed",
    }


def test_pipeline_explain_command_persists_run_and_emits_json_summary(
    tmp_path: Path,
) -> None:
    """The explain command should print a machine-readable SHAP summary."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-explain.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
        default_horizons=[1, 5, 20],
        min_training_days=80,
    )
    create_all_tables(database_url)
    ingest_training_history(settings)
    dataset = FeaturePipeline(settings=settings).build_dataset(date(2023, 11, 30), ["AAPL"])
    trained_models = TrainingService(settings=settings).train(dataset.id, [5])
    champion_model = next(model for model in trained_models if model.champion_rank == 1)

    stdout = StringIO()
    exit_code = main(
        [
            "explain",
            "--model-version-id",
            champion_model.id,
            "--sample-size",
            "8",
            "--top-signals",
            "3",
        ],
        settings=settings,
        stdout=stdout,
    )

    assert exit_code == 0

    payload = json.loads(stdout.getvalue())
    assert payload["command"] == "explain"
    assert payload["status"] == "completed"
    assert payload["model_version_id"] == champion_model.id
    assert payload["sample_size"] == 8
    assert payload["global_importance_count"] > 0
    assert payload["local_explanations_count"] > 0
    assert Path(payload["artifact_path"]).exists()

    with session_scope(database_url) as session:
        shap_run = session.scalar(select(ShapRun).where(ShapRun.id == payload["shap_run_id"]))

    assert shap_run is not None
    assert shap_run.model_version_id == champion_model.id
    assert shap_run.sample_size == payload["sample_size"]
    assert shap_run.artifact_path == payload["artifact_path"]
    assert shap_run.artifact_hash == payload["artifact_hash"]
    assert len(shap_run.summary_json["global_importance"]) == payload["global_importance_count"]
    assert len(shap_run.summary_json["local_explanations"]) == payload["local_explanations_count"]


def test_pipeline_explain_command_returns_nonzero_for_unknown_model_id(
    tmp_path: Path,
) -> None:
    """The explain command should fail cleanly for an unknown model version."""

    database_url = f"sqlite+pysqlite:///{tmp_path / 'pipeline-explain-failed.sqlite3'}"
    settings = Settings(
        database_url=database_url,
        artifact_root=tmp_path / "artifacts",
        benchmark_symbol="SPY",
        universe_symbols=["AAPL", "SPY"],
    )
    create_all_tables(database_url)

    stdout = StringIO()
    stderr = StringIO()
    exit_code = main(
        [
            "explain",
            "--model-version-id",
            "missing-model-version",
        ],
        settings=settings,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert stdout.getvalue() == ""
    assert json.loads(stderr.getvalue()) == {
        "command": "explain",
        "error": "Unknown model version: missing-model-version",
        "error_type": "ValueError",
        "status": "failed",
    }
