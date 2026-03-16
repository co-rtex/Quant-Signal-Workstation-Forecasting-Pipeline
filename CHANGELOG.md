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

### Validated

- `make lint`
- `make typecheck`
- `make test`
- `make db-up`
- `make migrate`
- synthetic ingestion and dataset materialization integration test
- training-to-API integration test covering model persistence and signal serving

### Next

- Database schema, migrations, and persistence contracts
- Market data ingestion workflow
- Feature datasets, model training, and signal serving
