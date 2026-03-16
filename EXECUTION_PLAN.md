# Execution Plan

## Objective

Deliver a production-minded MVP for daily US equities forecasting with reproducible datasets, calibrated multi-horizon models, regime-aware backtests, SHAP explainability, and FastAPI signal serving.

## Phase Status

| Phase | Status | Notes |
| --- | --- | --- |
| 1. Repo bootstrap and workflow | Complete | Standalone repo, package scaffold, docs, CI, local workflow |
| 2. Core config and storage | Pending | SQLAlchemy models, sessions, Alembic, registry schema |
| 3. Ingestion pipeline | Pending | Provider abstraction, yfinance adapter, ingestion runs, bar persistence |
| 4. Feature datasets | Pending | Feature engineering, labels, versioned Parquet datasets, split utilities |
| 5. Training and evaluation | Pending | Candidate models, calibration, champion selection, registry writes |
| 6. Backtesting and explainability | Pending | Walk-forward backtests, regime slices, SHAP artifacts |
| 7. FastAPI serving | Pending | Ranked signals, model metadata, readiness checks |

## Sequencing Rules

1. Finish the developer workflow before layering domain logic.
2. Add persistence contracts before writing pipeline orchestration.
3. Keep interfaces explicit so later steps do not rewrite earlier modules.
4. Validate after each meaningful change with lint, type checks, tests, and a focused runtime smoke check.

## Current Task

Implement configuration, database connectivity, SQLAlchemy models, and Alembic migrations.

## Next Task

Add provider abstraction, ingestion orchestration, and persisted raw bar storage.

## Key Risks

- Package compatibility on Python `3.14`
- Reproducibility drift if artifact metadata and database metadata diverge
- Leakage risk if time-aware dataset generation and validation are not enforced centrally

## Deferred Decisions

- Production market data provider beyond the free dev adapter
- Deployment target beyond local and CI workflow
- Hyperparameter search infrastructure beyond baseline champion selection
