"""Read-only signal and model metadata services."""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant_signal.core.config import Settings, get_settings
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import ModelEvaluation, ModelVersion, SignalSnapshot
from quant_signal.storage.repositories import StorageRepository


def rank_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank signals within each date by descending score."""

    ranked = frame.copy()
    ranked["rank"] = ranked.groupby("date")["score"].rank(method="first", ascending=False)
    ranked["rank"] = ranked["rank"].astype(int)
    return ranked.sort_values(["date", "rank"]).reset_index(drop=True)


class SignalService:
    """Read services for ranked signals, models, and evaluations."""

    def __init__(
        self,
        settings: Settings | None = None,
        database_url: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database_url = database_url or self.settings.database_url

    def get_ranked_signals(
        self,
        as_of_date: date,
        horizon: int,
        limit: int,
        model_version_id: str | None = None,
    ) -> list[SignalSnapshot]:
        """Return ranked persisted signals for a date and horizon."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            return repository.get_ranked_signal_snapshots(
                as_of_date=as_of_date,
                horizon_days=horizon,
                limit=limit,
                model_version_id=model_version_id,
            )

    def get_model_version(self, model_version_id: str) -> ModelVersion:
        """Return model metadata."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            return repository.get_model_version(model_version_id)

    def get_model_evaluations(self, model_version_id: str) -> list[ModelEvaluation]:
        """Return persisted evaluation splits for a model."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            return repository.list_model_evaluations(model_version_id)
