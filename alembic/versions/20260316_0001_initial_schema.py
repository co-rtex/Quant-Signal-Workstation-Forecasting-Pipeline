"""Initial schema for quant signal workstation."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260316_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the initial application schema."""

    op.create_table(
        "symbols",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("exchange", sa.String(length=64), nullable=True),
        sa.Column("asset_type", sa.String(length=32), nullable=False),
        sa.Column("is_benchmark", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_symbols"),
        sa.UniqueConstraint("symbol", name="uq_symbols_symbol"),
    )
    op.create_index("ix_symbols_symbol", "symbols", ["symbol"], unique=False)

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("requested_symbols", sa.JSON(), nullable=False),
        sa.Column("records_written", sa.Integer(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_ingestion_runs"),
    )

    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("feature_set_version", sa.String(length=64), nullable=False),
        sa.Column("horizons", sa.JSON(), nullable=False),
        sa.Column("symbols", sa.JSON(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("artifact_path", sa.String(length=512), nullable=False),
        sa.Column("artifact_hash", sa.String(length=64), nullable=False),
        sa.Column("feature_columns", sa.JSON(), nullable=False),
        sa.Column("label_columns", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_dataset_versions"),
    )

    op.create_table(
        "daily_bars",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("symbol_id", sa.String(length=36), nullable=False),
        sa.Column("ingestion_run_id", sa.String(length=36), nullable=True),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("adjusted_close", sa.Float(), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["ingestion_run_id"],
            ["ingestion_runs.id"],
            name="fk_daily_bars_ingestion_run_id_ingestion_runs",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["symbol_id"],
            ["symbols.id"],
            name="fk_daily_bars_symbol_id_symbols",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_daily_bars"),
        sa.UniqueConstraint(
            "symbol_id",
            "trade_date",
            "source",
            name="uq_daily_bars_symbol_date_source",
        ),
    )
    op.create_index("ix_daily_bars_trade_date", "daily_bars", ["trade_date"], unique=False)
    op.create_index(
        "ix_daily_bars_symbol_id_trade_date",
        "daily_bars",
        ["symbol_id", "trade_date"],
        unique=False,
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("model_family", sa.String(length=64), nullable=False),
        sa.Column("target_column", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("artifact_path", sa.String(length=512), nullable=False),
        sa.Column("artifact_hash", sa.String(length=64), nullable=False),
        sa.Column("feature_columns", sa.JSON(), nullable=False),
        sa.Column("champion_rank", sa.Integer(), nullable=True),
        sa.Column("train_start_date", sa.Date(), nullable=True),
        sa.Column("train_end_date", sa.Date(), nullable=True),
        sa.Column("validation_start_date", sa.Date(), nullable=True),
        sa.Column("validation_end_date", sa.Date(), nullable=True),
        sa.Column("test_start_date", sa.Date(), nullable=True),
        sa.Column("test_end_date", sa.Date(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            name="fk_model_versions_dataset_version_id_dataset_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_versions"),
    )
    op.create_index(
        "ix_model_versions_dataset_version_id",
        "model_versions",
        ["dataset_version_id"],
        unique=False,
    )

    op.create_table(
        "model_evaluations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("split_name", sa.String(length=32), nullable=False),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("pr_auc", sa.Float(), nullable=True),
        sa.Column("brier_score", sa.Float(), nullable=True),
        sa.Column("calibration_error", sa.Float(), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column("calibration_bins", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_model_evaluations_model_version_id_model_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_evaluations"),
        sa.UniqueConstraint(
            "model_version_id",
            "split_name",
            name="uq_model_evaluations_model_version_split",
        ),
    )

    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("top_n", sa.Integer(), nullable=False),
        sa.Column("min_training_days", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("artifact_path", sa.String(length=512), nullable=False),
        sa.Column("artifact_hash", sa.String(length=64), nullable=False),
        sa.Column("summary_json", sa.JSON(), nullable=False),
        sa.Column("regime_summary_json", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_backtest_runs_model_version_id_model_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_backtest_runs"),
    )

    op.create_table(
        "shap_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("artifact_path", sa.String(length=512), nullable=False),
        sa.Column("artifact_hash", sa.String(length=64), nullable=False),
        sa.Column("summary_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_shap_runs_model_version_id_model_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_shap_runs"),
    )

    op.create_table(
        "signal_snapshots",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name="fk_signal_snapshots_model_version_id_model_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_signal_snapshots"),
        sa.UniqueConstraint(
            "model_version_id",
            "as_of_date",
            "symbol",
            name="uq_signal_snapshots_model_date_symbol",
        ),
    )
    op.create_index(
        "ix_signal_snapshots_as_of_date_horizon_days",
        "signal_snapshots",
        ["as_of_date", "horizon_days"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the initial application schema."""

    op.drop_index("ix_signal_snapshots_as_of_date_horizon_days", table_name="signal_snapshots")
    op.drop_table("signal_snapshots")
    op.drop_table("shap_runs")
    op.drop_table("backtest_runs")
    op.drop_table("model_evaluations")
    op.drop_index("ix_model_versions_dataset_version_id", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index("ix_daily_bars_symbol_id_trade_date", table_name="daily_bars")
    op.drop_index("ix_daily_bars_trade_date", table_name="daily_bars")
    op.drop_table("daily_bars")
    op.drop_table("dataset_versions")
    op.drop_table("ingestion_runs")
    op.drop_index("ix_symbols_symbol", table_name="symbols")
    op.drop_table("symbols")
