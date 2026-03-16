"""Feature dataset materialization pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from quant_signal.core.config import Settings, get_settings
from quant_signal.core.hashing import sha256_file
from quant_signal.features.engineering import FEATURE_COLUMNS, build_feature_frame
from quant_signal.features.labels import (
    add_forward_return_targets,
    forward_return_columns,
    target_columns,
)
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import DatasetVersion
from quant_signal.storage.repositories import DatasetArtifactRecord, StorageRepository


class FeaturePipeline:
    """Build versioned, model-ready datasets from persisted market bars."""

    def __init__(
        self,
        settings: Settings | None = None,
        database_url: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database_url = database_url or self.settings.database_url

    def build_dataset(
        self,
        as_of_date: date,
        symbols: Sequence[str] | None = None,
        feature_set_version: str = "ohlcv_v1",
    ) -> DatasetVersion:
        """Materialize and register a dataset artifact."""

        benchmark_symbol = self.settings.benchmark_symbol.upper()
        requested = [symbol.upper() for symbol in (symbols or self.settings.universe_symbols)]
        modeling_symbols = sorted({symbol for symbol in requested if symbol != benchmark_symbol})
        fetch_symbols = sorted({*modeling_symbols, benchmark_symbol})

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            bars = repository.load_daily_bars_frame(fetch_symbols, end_date=as_of_date)

        if bars.empty:
            raise ValueError("No persisted bars were found for the requested dataset build.")

        feature_frame = build_feature_frame(bars, benchmark_symbol=benchmark_symbol)
        dataset_frame = feature_frame[feature_frame["symbol"].isin(modeling_symbols)].copy()
        dataset_frame = add_forward_return_targets(dataset_frame, self.settings.default_horizons)
        dataset_frame["date"] = pd.to_datetime(dataset_frame["date"])
        dataset_frame = dataset_frame.sort_values(["date", "symbol"]).reset_index(drop=True)

        label_columns = target_columns(self.settings.default_horizons) + forward_return_columns(
            self.settings.default_horizons
        )
        artifact_path = self._build_artifact_path(as_of_date, feature_set_version)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_frame.to_parquet(artifact_path, index=False)
        artifact_hash = sha256_file(artifact_path)

        metadata_json: dict[str, object] = {
            "date_range": {
                "start": str(dataset_frame["date"].min().date()),
                "end": str(dataset_frame["date"].max().date()),
            },
            "benchmark_symbol": benchmark_symbol,
            "raw_row_count": int(len(bars)),
        }

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            return repository.create_dataset_version(
                DatasetArtifactRecord(
                    as_of_date=as_of_date,
                    feature_set_version=feature_set_version,
                    horizons=list(self.settings.default_horizons),
                    symbols=modeling_symbols,
                    row_count=int(len(dataset_frame)),
                    artifact_path=str(artifact_path),
                    artifact_hash=artifact_hash,
                    feature_columns=FEATURE_COLUMNS,
                    label_columns=label_columns,
                    metadata_json=metadata_json,
                )
            )

    def _build_artifact_path(self, as_of_date: date, feature_set_version: str) -> Path:
        """Build a stable dataset artifact path."""

        filename = f"dataset_{feature_set_version}_{as_of_date:%Y%m%d}.parquet"
        return Path(self.settings.artifact_root) / "datasets" / filename
