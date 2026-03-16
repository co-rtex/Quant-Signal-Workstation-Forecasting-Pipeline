# Architecture

## Design Goals

- Reproducibility over speed
- Explicit boundaries between HTTP, orchestration, domain logic, and persistence
- Local developer ergonomics with containerized infrastructure and artifact versioning
- Boring, reviewable solutions that can expand without large rewrites

## System Boundaries

- `quant_signal.api`: FastAPI application, route handlers, request and response schemas
- `quant_signal.core`: configuration, hashing, logging, shared constants
- `quant_signal.storage`: SQLAlchemy models, database sessions, repository helpers
- `quant_signal.ingestion`: provider contract, development adapter, ingestion orchestration
- `quant_signal.features`: feature engineering, label generation, dataset versioning, split logic
- `quant_signal.training`: model training, calibration, evaluation orchestration, artifact persistence
- `quant_signal.evaluation`: metrics and reporting utilities
- `quant_signal.backtesting`: walk-forward simulation and regime analysis
- `quant_signal.explainability`: SHAP generation tied to model versions
- `quant_signal.serving`: ranked signal retrieval and model metadata read services

## Persistence Model

- PostgreSQL stores normalized raw bars, ingestion runs, dataset manifests, model registry records, evaluations, backtests, explainability run metadata, and signal snapshots.
- Versioned Parquet artifacts under `artifacts/` store model-ready feature datasets and serialized model bundles.
- Hashes and metadata in PostgreSQL point to the exact artifact path used by downstream stages.

## Data Flow

1. Ingestion fetches bars from a provider contract and stores normalized OHLCV rows with source and ingestion metadata.
2. Feature generation queries persisted bars, joins benchmark context, creates leakage-safe features and labels, and writes a versioned dataset artifact.
3. Training loads a dataset artifact, applies time-aware splits, fits candidate models, calibrates probabilities, evaluates reliability, and registers the champion model per horizon.
4. Backtesting retrains monthly in walk-forward mode, scores daily candidates, simulates overlapping horizon sleeves, and records performance summaries plus regime slices.
5. Explainability loads model bundles and evaluation slices, produces SHAP summaries, and stores explainability artifacts.
6. FastAPI serves health, model metadata, and ranked historical signal snapshots from persisted records.

## Initial Tradeoffs

- Use sync SQLAlchemy sessions for simplicity and testability.
- Keep training request handling out of the API; API remains read-only.
- Use a free adapter first, but hide it behind a provider interface to avoid leaking vendor assumptions.
- Store wide feature matrices in Parquet artifacts rather than a mutable wide SQL table.
