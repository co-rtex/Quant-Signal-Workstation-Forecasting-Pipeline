"""Alembic migration smoke tests."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_alembic_upgrade_head(tmp_path: Path) -> None:
    """Applying the head migration should create the core schema."""

    database_path = tmp_path / "migrations.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path}"

    project_root = Path(__file__).resolve().parents[2]
    alembic_config = Config(str(project_root / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(project_root / "alembic"))
    alembic_config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(alembic_config, "head")

    inspector = inspect(create_engine(database_url))
    assert {
        "symbols",
        "ingestion_runs",
        "daily_bars",
        "dataset_versions",
        "model_versions",
        "model_evaluations",
        "backtest_runs",
        "shap_runs",
        "signal_snapshots",
    }.issubset(set(inspector.get_table_names()))
