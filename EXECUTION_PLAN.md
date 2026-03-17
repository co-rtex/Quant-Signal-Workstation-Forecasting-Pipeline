# Execution Plan

## Objective

Extend the validated MVP with production-minded realism improvements, starting with cost-aware backtesting and then moving into richer analytics, provider hardening, and scheduled orchestration.

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
| 8. Backtest cost realism | Complete | Transaction costs, slippage, and auditable execution assumptions |
| 9. Benchmark-relative analytics | Complete | Active-return metrics, richer regime context, benchmark-relative drawdowns |
| 10. Turnover analytics foundation | Complete | Detail artifact, holdings transitions, turnover-aware reporting |
| 11. Attribution-ready analytics foundation | Complete | Detail-artifact benchmark contributions and lifecycle attribution summaries |
| 12. Regime-aware attribution analytics | Pending | Regime-sliced implementation diagnostics and grouped attribution summaries |

## Sequencing Rules

1. Finish the developer workflow before layering domain logic.
2. Add persistence contracts before writing pipeline orchestration.
3. Keep interfaces explicit so later steps do not rewrite earlier modules.
4. Validate after each meaningful change with lint, type checks, tests, and a focused runtime smoke check.

## Current Task

Plan the next post-attribution slice: regime-aware attribution summaries.

## Next Task

Regime-aware attribution summaries and benchmark attribution hooks.

## Key Risks

- Package compatibility on Python `3.14`
- Reproducibility drift if artifact metadata and database metadata diverge
- Leakage risk if time-aware dataset generation and validation are not enforced centrally
- Backtest analytics still exclude regime-sliced attribution, transaction timing nuance, and multi-strategy comparisons

## Deferred Decisions

- Production market data provider beyond the free dev adapter
- Deployment target beyond local and CI workflow
- Hyperparameter search infrastructure beyond baseline champion selection
- Benchmark attribution, provider hardening, and multi-strategy comparisons
