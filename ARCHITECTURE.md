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
- The v1 schema uses logically grouped tables in the default database schema to keep local testing portable; service boundaries do not depend on physical schema names, so a later Postgres schema split remains reversible.

## Data Flow

1. Ingestion fetches bars from a provider contract, records request/provider/provider-fetch metadata at the run level, and stores normalized OHLCV rows with source and ingestion metadata.
2. Feature generation queries persisted bars, joins benchmark context, creates leakage-safe features and labels, and writes a versioned dataset artifact.
3. Training loads a dataset artifact, applies time-aware splits, fits candidate models, calibrates probabilities, evaluates reliability, and registers the champion model per horizon.
4. Backtesting retrains monthly in walk-forward mode, scores daily candidates, simulates overlapping horizon sleeves, applies configurable transaction costs and slippage, and records gross/net performance summaries, benchmark-relative analytics, richer regime slices, turnover-aware detail artifacts, attribution-ready summaries, and regime-aware attribution summaries.
5. Explainability loads model bundles and evaluation slices, produces SHAP summaries, and stores explainability artifacts.
6. FastAPI serves health, model metadata, and ranked historical signal snapshots from persisted records.

## Contract Points

- `MarketDataProvider.fetch_daily_bars(symbols, start_date, end_date)` returns a `ProviderFetchResult` envelope with normalized daily bars, returned-symbol coverage, provider metadata, and warnings.
- `FeaturePipeline.build_dataset(as_of_date, symbols, feature_set_version)` owns feature and label materialization plus dataset manifest creation.
- `TrainingService.train(dataset_version_id, horizons)` owns candidate fitting, probability calibration, evaluation persistence, champion selection, and signal snapshot refresh.
- `BacktestService.run(model_version_id, top_n)` retrains the registered model family in a monthly walk-forward loop and persists summary artifacts.
- `BacktestService.run(model_version_id, top_n, execution_assumptions)` supports reproducible cost-aware reruns without changing the database schema.
- `label_regimes(benchmark_frame)` preserves the primary trend/volatility regime while adding benchmark momentum and drawdown context for grouped analytics.
- Backtest runs now emit a primary daily artifact plus a companion detail artifact referenced from `metadata_json`, which keeps composition-level reporting schema-light and reversible.
- Attribution-ready reporting is derived from the detail artifact, which carries benchmark-relative contribution fields and lifecycle flags without changing the backtest database schema.
- Regime-aware attribution reporting stays daily-grain: primary regime summaries live in `regime_summary_json`, while dimension-based attribution summaries live under `summary_json`.
- `ExplainabilityService.generate(model_version_id, sample_size, top_signals)` binds SHAP outputs to a specific registered model artifact and evaluation window.
- `SignalService.get_ranked_signals(as_of_date, horizon, limit)` is the read-side contract used by the API layer.
- Ingestion run metadata stays schema-light and nested inside `metadata_json`: request parameters, provider configuration, provider fetch diagnostics, and persistence counts are persisted without widening the database schema.

## Initial Tradeoffs

- Use sync SQLAlchemy sessions for simplicity and testability.
- Keep training request handling out of the API; API remains read-only.
- Use a free adapter first, but hide it behind a provider interface to avoid leaking vendor assumptions.
- Keep provider selection config-driven so scheduled entrypoints can become thin orchestration later.
- Store wide feature matrices in Parquet artifacts rather than a mutable wide SQL table.
- Keep backtesting assumptions explicit and simple: monthly retraining, equal-weight long-only sleeves, benchmark-derived regime labels, configurable per-side transaction cost/slippage assumptions, benchmark-relative analytics, and companion detail artifacts for turnover-aware, attribution-ready, and regime-aware reporting.
