# Execution Plan

## Objective

Deliver a production-minded MVP for daily US equities forecasting with reproducible datasets, calibrated multi-horizon models, regime-aware backtests, SHAP explainability, and FastAPI signal serving.

## Phase Status

| Phase | Status | Notes |
| --- | --- | --- |
| 1. Repo bootstrap and workflow | Complete | Standalone repo, package scaffold, docs, CI, local workflow |
| 2. Core config and storage | Complete | SQLAlchemy models, readiness checks, Alembic, registry schema |
| 3. Ingestion pipeline | Complete | Provider abstraction, yfinance adapter, ingestion runs, bar persistence |
| 4. Feature datasets | Complete | Feature engineering, labels, versioned Parquet datasets, split utilities |
| 5. Training and evaluation | Complete | Candidate models, calibration, champion selection, registry writes |
| 6. Backtesting and explainability | Complete | Monthly walk-forward backtests, regime slices, SHAP artifacts |
| 7. FastAPI serving | Complete | Ranked signals, model metadata, readiness checks |

## Sequencing Rules

1. Finish the developer workflow before layering domain logic.
2. Add persistence contracts before writing pipeline orchestration.
3. Keep interfaces explicit so later steps do not rewrite earlier modules.
4. Validate after each meaningful change with lint, type checks, tests, and a focused runtime smoke check.

## Current Task

Refresh documentation, run final validation, and prepare the clean Git history for push.

## Next Task

Push the validated MVP scaffold and pipeline implementation to GitHub.

## Key Risks

- Package compatibility on Python `3.14`
- Reproducibility drift if artifact metadata and database metadata diverge
- Leakage risk if time-aware dataset generation and validation are not enforced centrally
- Backtest realism remains intentionally simple for MVP and should evolve before any live use

## Deferred Decisions

- Production market data provider beyond the free dev adapter
- Deployment target beyond local and CI workflow
- Hyperparameter search infrastructure beyond baseline champion selection
- Transaction cost, slippage, and short-side portfolio modeling in backtests
