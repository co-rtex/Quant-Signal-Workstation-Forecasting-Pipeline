# Changelog

## Unreleased

### Added

- Standalone repository bootstrap for the quant forecasting platform
- Project packaging, developer workflow, CI scaffold, and environment configuration
- Architecture notes, execution plan, and delivery log
- FastAPI health endpoints and basic runtime configuration
- SQLAlchemy models, database helpers, Alembic environment, and the initial persistence schema
- Readiness checks backed by real database connectivity
- Provider abstraction, ingestion service, and persisted normalized daily bar workflow
- Feature engineering, forward-return label generation, temporal split utilities, and dataset artifact materialization
- Baseline model training, Platt-style calibration, evaluation metrics, and champion selection rules
- Persisted model registry records, signal snapshots, and read-only FastAPI endpoints for signals and model metadata
- Walk-forward backtesting with benchmark regime summaries and persisted backtest artifacts
- Configurable transaction cost and slippage modeling for walk-forward backtests
- Benchmark-relative backtest analytics with richer momentum and drawdown regime context
- Turnover-aware backtest reporting with companion detail artifacts
- Attribution-ready backtest summaries built from benchmark-relative detail contributions
- Regime-aware attribution summaries for primary regimes and grouped regime dimensions
- Provider fetch envelopes, config-driven provider selection, and richer ingestion run metadata
- SHAP explainability workflow with persisted global and local explanation artifacts tied to model versions

### Validated

- `make lint`
- `make typecheck`
- `make test`
- `make db-up`
- `make migrate`
- synthetic ingestion and dataset materialization integration test
- training-to-API integration test covering model persistence and signal serving
- backtesting and explainability integration test covering persisted artifacts
- cost-aware backtesting unit and integration coverage
- benchmark-relative analytics and richer regime coverage for backtests
- turnover analytics unit and integration coverage for backtests
- attribution-ready detail summary coverage for backtests
- regime-aware attribution summary coverage for backtests
- provider metadata/configuration coverage for ingestion

### Next

- Retry-aware market data ingestion
- Scheduled pipeline entrypoints
