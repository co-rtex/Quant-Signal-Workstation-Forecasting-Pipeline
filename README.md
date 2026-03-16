# Quant Signal Workstation & Forecasting Pipeline

Production-minded forecasting platform for daily US equities. The system ingests market data, builds reproducible feature datasets, trains calibrated multi-horizon models, runs regime-aware backtests, generates SHAP explainability artifacts, and serves ranked signals through FastAPI.

## MVP Scope

- Daily US equities universe with multi-horizon binary targets for `1D`, `5D`, and `20D`
- Replaceable market data provider contract with a free development adapter first
- PostgreSQL-backed metadata registry and normalized market data storage
- Versioned Parquet feature datasets under `artifacts/`
- Calibrated probabilistic models, evaluation reporting, walk-forward backtests, and SHAP outputs
- Read-only FastAPI endpoints for health, model metadata, and ranked signal snapshots

## Repository Layout

```text
.
├── alembic/                  # Database migrations
├── artifacts/                # Local artifact storage (gitignored)
├── src/quant_signal/         # Application package
├── tests/                    # Unit and integration tests
├── ARCHITECTURE.md           # System design and tradeoffs
├── CHANGELOG.md              # Running delivery log
├── EXECUTION_PLAN.md         # Living execution plan
├── Makefile                  # Local developer workflow
├── docker-compose.yml        # Local PostgreSQL
└── pyproject.toml            # Packaging, lint, type, and test config
```

## Quick Start

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .[dev]
   ```

2. Copy environment defaults:

   ```bash
   cp .env.example .env
   ```

3. Start PostgreSQL:

   ```bash
   docker compose up -d postgres
   ```

4. Run migrations:

   ```bash
   make migrate
   ```

5. Start the API:

   ```bash
   make run-api
   ```

## Validation Workflow

- `make lint`
- `make typecheck`
- `make test`
- `make validate`

## Current Status

The repository now includes the validated platform foundation, database schema, ingestion contract, persisted OHLCV workflow, feature engineering pipeline, versioned dataset artifacts, baseline model training, probability calibration, persisted signal snapshots, and read-only FastAPI endpoints for health, signals, and model metadata. The remaining major implementation slice is walk-forward backtesting plus SHAP explainability tied to registered model versions.
