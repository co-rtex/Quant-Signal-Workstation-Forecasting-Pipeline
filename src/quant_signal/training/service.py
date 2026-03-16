"""Training, calibration, evaluation, and registry persistence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant_signal.core.config import Settings, get_settings
from quant_signal.core.hashing import sha256_file, sha256_json
from quant_signal.evaluation.metrics import compute_classification_metrics
from quant_signal.evaluation.reporting import rank_candidate_metrics
from quant_signal.features.splits import TemporalSplit, build_temporal_split
from quant_signal.serving.service import rank_signal_frame
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import ModelVersion
from quant_signal.storage.repositories import (
    EvaluationRecord,
    ModelArtifactRecord,
    SignalSnapshotRecord,
    StorageRepository,
)
from quant_signal.training.artifacts import ModelArtifactBundle, ProbabilityCalibrator


@dataclass
class CandidateTrainingResult:
    """In-memory training result for a candidate model family."""

    model_family: str
    bundle: ModelArtifactBundle
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    artifact_path: Path
    artifact_hash: str


class TrainingService:
    """Train calibrated model families and persist registry outputs."""

    def __init__(
        self,
        settings: Settings | None = None,
        database_url: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database_url = database_url or self.settings.database_url

    def train(
        self,
        dataset_version_id: str,
        horizons: Sequence[int] | None = None,
    ) -> list[ModelVersion]:
        """Train and persist candidate models for the requested horizons."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            dataset_version = repository.get_dataset_version(dataset_version_id)

        dataset_frame = pd.read_parquet(dataset_version.artifact_path)
        dataset_frame["date"] = pd.to_datetime(dataset_frame["date"])
        requested_horizons = list(horizons or dataset_version.horizons)
        persisted_models: list[ModelVersion] = []

        for horizon in requested_horizons:
            target_column = f"target_up_{horizon}d"

            training_frame = dataset_frame.dropna(subset=[target_column]).copy()
            split = build_temporal_split(
                training_frame,
                embargo_days=horizon,
                minimum_train_dates=self.settings.min_training_days,
            )
            feature_columns = list(dataset_version.feature_columns)
            candidate_results = self._train_candidates(
                dataset_frame=training_frame,
                dataset_version_id=dataset_version.id,
                feature_columns=feature_columns,
                target_column=target_column,
                horizon=horizon,
                split=split,
            )

            ranked_candidates = rank_candidate_metrics(
                [
                    {
                        "model_family": result.model_family,
                        "validation_metrics": result.validation_metrics,
                    }
                    for result in candidate_results
                ]
            )
            ranking_map = {
                record["model_family"]: index
                for index, record in enumerate(ranked_candidates, start=1)
            }

            with session_scope(self.database_url) as session:
                repository = StorageRepository(session)
                for result in candidate_results:
                    champion_rank = ranking_map[result.model_family]
                    split_summary = result.bundle.split_summary
                    train_summary = split_summary["train"]
                    validation_summary = split_summary["validation"]
                    test_summary = split_summary["test"]
                    model_version = repository.create_model_version(
                        ModelArtifactRecord(
                            dataset_version_id=dataset_version.id,
                            horizon_days=horizon,
                            model_family=result.model_family,
                            target_column=target_column,
                            artifact_path=str(result.artifact_path),
                            artifact_hash=result.artifact_hash,
                            feature_columns=feature_columns,
                            champion_rank=champion_rank,
                            train_start_date=cast(date | None, train_summary["start_date"]),
                            train_end_date=cast(date | None, train_summary["end_date"]),
                            validation_start_date=cast(
                                date | None,
                                validation_summary["start_date"],
                            ),
                            validation_end_date=cast(date | None, validation_summary["end_date"]),
                            test_start_date=cast(date | None, test_summary["start_date"]),
                            test_end_date=cast(date | None, test_summary["end_date"]),
                            metadata_json=result.bundle.metadata_json,
                        )
                    )
                    repository.create_model_evaluation(
                        EvaluationRecord(
                            model_version_id=model_version.id,
                            split_name="validation",
                            metrics_json=result.validation_metrics,
                            calibration_bins=result.validation_metrics["calibration_bins"],
                        )
                    )
                    repository.create_model_evaluation(
                        EvaluationRecord(
                            model_version_id=model_version.id,
                            split_name="test",
                            metrics_json=result.test_metrics,
                            calibration_bins=result.test_metrics["calibration_bins"],
                        )
                    )

                    if champion_rank == 1:
                        snapshots = self._build_signal_snapshots(
                            model_version_id=model_version.id,
                            bundle=result.bundle,
                            dataset_frame=dataset_frame,
                        )
                        repository.replace_signal_snapshots(model_version.id, snapshots)

                    persisted_models.append(model_version)

        return persisted_models

    def _train_candidates(
        self,
        dataset_frame: pd.DataFrame,
        dataset_version_id: str,
        feature_columns: list[str],
        target_column: str,
        horizon: int,
        split: TemporalSplit,
    ) -> list[CandidateTrainingResult]:
        """Train all baseline candidate model families for a horizon."""

        train_frame = dataset_frame[split.mask(dataset_frame, "train")].copy()
        validation_frame = dataset_frame[split.mask(dataset_frame, "validation")].copy()
        test_frame = dataset_frame[split.mask(dataset_frame, "test")].copy()

        x_train = train_frame[feature_columns]
        x_validation = validation_frame[feature_columns]
        x_test = test_frame[feature_columns]
        y_train = train_frame[target_column].astype(int).to_numpy()
        y_validation = validation_frame[target_column].astype(int).to_numpy()
        y_test = test_frame[target_column].astype(int).to_numpy()

        self._ensure_class_diversity(y_train, split_name="train", horizon=horizon)
        self._ensure_class_diversity(y_validation, split_name="validation", horizon=horizon)
        self._ensure_class_diversity(y_test, split_name="test", horizon=horizon)

        split_summary = self._build_split_summary(train_frame, validation_frame, test_frame)
        results: list[CandidateTrainingResult] = []

        for model_family, estimator in self._candidate_estimators().items():
            fitted_estimator = estimator.fit(x_train, y_train)
            validation_raw = fitted_estimator.predict_proba(x_validation)[:, 1]
            calibrator = ProbabilityCalibrator.fit(validation_raw, y_validation)
            validation_probabilities = calibrator.predict(validation_raw)
            test_probabilities = calibrator.predict(fitted_estimator.predict_proba(x_test)[:, 1])

            validation_metrics = compute_classification_metrics(
                y_validation,
                validation_probabilities,
            )
            test_metrics = compute_classification_metrics(y_test, test_probabilities)

            bundle = ModelArtifactBundle(
                model_family=model_family,
                feature_columns=feature_columns,
                target_column=target_column,
                horizon_days=horizon,
                dataset_version_id=dataset_version_id,
                estimator=fitted_estimator,
                calibrator=calibrator,
                split_summary=split_summary,
                metadata_json={
                    "trained_at": datetime.now(tz=UTC).isoformat(),
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                },
            )
            artifact_path = self._build_artifact_path(
                dataset_version_id,
                horizon,
                model_family,
                split_summary,
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, artifact_path)

            results.append(
                CandidateTrainingResult(
                    model_family=model_family,
                    bundle=bundle,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    artifact_path=artifact_path,
                    artifact_hash=sha256_file(artifact_path),
                )
            )

        return results

    def _build_signal_snapshots(
        self,
        model_version_id: str,
        bundle: ModelArtifactBundle,
        dataset_frame: pd.DataFrame,
    ) -> list[SignalSnapshotRecord]:
        """Score and rank the full dataset for API-ready signal snapshots."""

        scorable = dataset_frame.copy()
        feature_coverage = scorable[bundle.feature_columns].notna().sum(axis=1)
        scorable = scorable[feature_coverage >= max(5, len(bundle.feature_columns) // 2)].copy()
        scorable["score"] = bundle.predict_positive_proba(scorable)
        ranked = rank_signal_frame(scorable[["date", "symbol", "score"]])

        return [
            SignalSnapshotRecord(
                model_version_id=model_version_id,
                horizon_days=bundle.horizon_days,
                as_of_date=row.date.date(),
                symbol=str(row.symbol),
                score=float(row.score),
                rank=int(row.rank),
                metadata_json={"model_family": bundle.model_family},
            )
            for row in ranked.itertuples(index=False)
        ]

    def _candidate_estimators(self) -> dict[str, Pipeline]:
        """Return the baseline candidate model families."""

        return {
            "logistic_regression": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
            "hist_gradient_boosting": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_depth=4,
                            learning_rate=0.05,
                            max_iter=250,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        }

    def _ensure_class_diversity(
        self,
        targets: Any,
        split_name: str,
        horizon: int,
    ) -> None:
        """Raise a clear error when a split lacks both target classes."""

        unique_classes = set(targets.tolist())
        if len(unique_classes) < 2:
            raise ValueError(
                f"Horizon {horizon}d has only one target class in the {split_name} split."
            )

    def _build_artifact_path(
        self,
        dataset_version_id: str,
        horizon: int,
        model_family: str,
        split_summary: dict[str, dict[str, Any]],
    ) -> Path:
        """Build a deterministic artifact path for a model candidate."""

        artifact_key = sha256_json(
            {
                "dataset_version_id": dataset_version_id,
                "horizon": horizon,
                "model_family": model_family,
                "split_summary": split_summary,
            }
        )[:12]
        filename = f"model_{dataset_version_id[:8]}_{horizon}d_{model_family}_{artifact_key}.joblib"
        return Path(self.settings.artifact_root) / "models" / filename

    def _build_split_summary(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """Serialize split boundaries for artifact metadata."""

        return {
            "train": self._frame_summary(train_frame),
            "validation": self._frame_summary(validation_frame),
            "test": self._frame_summary(test_frame),
        }

    def _frame_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Return a date-window summary for a split frame."""

        return {
            "start_date": frame["date"].min().date(),
            "end_date": frame["date"].max().date(),
            "row_count": int(len(frame)),
        }
