"""ORM models for persisted market, dataset, model, and signal metadata."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from quant_signal.core.time import utc_now
from quant_signal.storage.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Symbol(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Tracked symbol metadata."""

    __tablename__ = "symbols"

    symbol: Mapped[str] = mapped_column(String(16), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    exchange: Mapped[str | None] = mapped_column(String(64), nullable=True)
    asset_type: Mapped[str] = mapped_column(String(32), default="equity", nullable=False)
    is_benchmark: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    daily_bars: Mapped[list[DailyBar]] = relationship(back_populates="symbol")


class IngestionRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Market data ingestion run metadata."""

    __tablename__ = "ingestion_runs"

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    requested_symbols: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    records_written: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    metadata_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    bars: Mapped[list[DailyBar]] = relationship(back_populates="ingestion_run")


class DailyBar(UUIDPrimaryKeyMixin, Base):
    """Normalized daily OHLCV market bar."""

    __tablename__ = "daily_bars"

    symbol_id: Mapped[str] = mapped_column(
        ForeignKey("symbols.id", ondelete="CASCADE"),
        nullable=False,
    )
    ingestion_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    trade_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    symbol: Mapped[Symbol] = relationship(back_populates="daily_bars")
    ingestion_run: Mapped[IngestionRun | None] = relationship(back_populates="bars")


class DatasetVersion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Dataset artifact metadata."""

    __tablename__ = "dataset_versions"

    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    feature_set_version: Mapped[str] = mapped_column(String(64), nullable=False)
    horizons: Mapped[list[int]] = mapped_column(JSON, default=list, nullable=False)
    symbols: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_columns: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    label_columns: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    metadata_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict, nullable=False)

    model_versions: Mapped[list[ModelVersion]] = relationship(back_populates="dataset_version")


class ModelVersion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Trained model artifact metadata."""

    __tablename__ = "model_versions"

    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    model_family: Mapped[str] = mapped_column(String(64), nullable=False)
    target_column: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="trained", nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_columns: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    champion_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    train_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    train_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    validation_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    validation_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    test_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    test_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    metadata_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict, nullable=False)

    dataset_version: Mapped[DatasetVersion] = relationship(back_populates="model_versions")
    evaluations: Mapped[list[ModelEvaluation]] = relationship(back_populates="model_version")
    backtest_runs: Mapped[list[BacktestRun]] = relationship(back_populates="model_version")
    shap_runs: Mapped[list[ShapRun]] = relationship(back_populates="model_version")
    signal_snapshots: Mapped[list[SignalSnapshot]] = relationship(back_populates="model_version")


class ModelEvaluation(UUIDPrimaryKeyMixin, Base):
    """Persisted evaluation metrics for a model split."""

    __tablename__ = "model_evaluations"

    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    split_name: Mapped[str] = mapped_column(String(32), nullable=False)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    pr_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict, nullable=False)
    calibration_bins: Mapped[list[dict[str, float]]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    model_version: Mapped[ModelVersion] = relationship(back_populates="evaluations")


class BacktestRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Walk-forward backtest metadata."""

    __tablename__ = "backtest_runs"

    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    top_n: Mapped[int] = mapped_column(Integer, nullable=False)
    min_training_days: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    summary_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict, nullable=False)
    regime_summary_json: Mapped[dict[str, object]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    metadata_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model_version: Mapped[ModelVersion] = relationship(back_populates="backtest_runs")


class ShapRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """SHAP explainability artifact metadata."""

    __tablename__ = "shap_runs"

    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    summary_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict, nullable=False)

    model_version: Mapped[ModelVersion] = relationship(back_populates="shap_runs")


class SignalSnapshot(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Persisted ranked signal output for a model and date."""

    __tablename__ = "signal_snapshots"

    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict[str, object]] = mapped_column(JSON, default=dict, nullable=False)

    model_version: Mapped[ModelVersion] = relationship(back_populates="signal_snapshots")
