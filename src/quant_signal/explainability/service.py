"""SHAP explainability generation tied to model versions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from quant_signal.core.config import Settings, get_settings
from quant_signal.core.hashing import sha256_file
from quant_signal.storage.db import session_scope
from quant_signal.storage.models import ShapRun
from quant_signal.storage.repositories import ShapRunRecord, StorageRepository


class ExplainabilityService:
    """Generate reproducible SHAP outputs for a persisted model version."""

    def __init__(
        self,
        settings: Settings | None = None,
        database_url: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database_url = database_url or self.settings.database_url

    def generate(
        self,
        model_version_id: str,
        sample_size: int = 32,
        top_signals: int = 5,
    ) -> ShapRun:
        """Generate and persist SHAP summary artifacts for a model version."""

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            model_version = repository.get_model_version(model_version_id)
            dataset_version = repository.get_dataset_version(model_version.dataset_version_id)

        bundle = joblib.load(model_version.artifact_path)
        dataset_frame = pd.read_parquet(dataset_version.artifact_path)
        dataset_frame["date"] = pd.to_datetime(dataset_frame["date"])
        evaluation_frame = dataset_frame[
            dataset_frame["date"].between(
                pd.Timestamp(model_version.test_start_date),
                pd.Timestamp(model_version.test_end_date),
            )
        ].copy()
        if evaluation_frame.empty:
            raise ValueError(
                f"No evaluation observations available for model version {model_version_id}."
            )

        sample = evaluation_frame.sample(
            n=min(sample_size, len(evaluation_frame)),
            random_state=42,
        ).reset_index(drop=True)
        background = evaluation_frame.head(min(25, len(evaluation_frame))).copy()
        feature_frame = sample[bundle.feature_columns]
        background_frame = background[bundle.feature_columns]

        def predict_probability(values: np.ndarray) -> np.ndarray:
            probability_frame = pd.DataFrame(values, columns=bundle.feature_columns)
            return np.asarray(
                bundle.predict_positive_proba(probability_frame),
                dtype=float,
            )

        explainer = shap.Explainer(predict_probability, background_frame.to_numpy())
        shap_values = explainer(feature_frame.to_numpy())
        shap_matrix = np.asarray(shap_values.values, dtype=float)
        mean_abs_values = np.mean(np.abs(shap_matrix), axis=0)
        global_importance = sorted(
            [
                {"feature": feature, "mean_abs_shap": float(score)}
                for feature, score in zip(bundle.feature_columns, mean_abs_values, strict=False)
            ],
            key=lambda row: row["mean_abs_shap"],
            reverse=True,
        )

        sample_with_scores = sample[["date", "symbol"]].copy()
        sample_with_scores["score"] = bundle.predict_positive_proba(sample)
        top_local_rows = sample_with_scores.sort_values("score", ascending=False).head(top_signals)
        local_explanations: list[dict[str, object]] = []
        for row_index, row in top_local_rows.iterrows():
            contributions = sorted(
                [
                    {"feature": feature, "shap_value": float(shap_matrix[row_index, feature_index])}
                    for feature_index, feature in enumerate(bundle.feature_columns)
                ],
                key=lambda item: abs(item["shap_value"]),
                reverse=True,
            )[:5]
            local_explanations.append(
                {
                    "date": str(pd.Timestamp(row["date"]).date()),
                    "symbol": str(row["symbol"]),
                    "score": float(row["score"]),
                    "top_contributors": contributions,
                }
            )

        summary_json: dict[str, object] = {
            "global_importance": global_importance,
            "local_explanations": local_explanations,
            "sample_size": int(len(sample)),
        }

        artifact_path = (
            Path(self.settings.artifact_root)
            / "explainability"
            / f"shap_{model_version_id}.json"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(summary_json, indent=2, default=str),
            encoding="utf-8",
        )

        with session_scope(self.database_url) as session:
            repository = StorageRepository(session)
            return repository.create_shap_run(
                ShapRunRecord(
                    model_version_id=model_version_id,
                    sample_size=int(len(sample)),
                    artifact_path=str(artifact_path),
                    artifact_hash=sha256_file(artifact_path),
                    summary_json=summary_json,
                )
            )
