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

6. Run a scheduled-friendly ingestion command:

   ```bash
   quant-signal-pipeline ingest --start-date 2024-01-02 --end-date 2024-05-31 --symbols AAPL
   ```

7. Build a versioned dataset artifact:

   ```bash
   quant-signal-pipeline build-dataset --as-of-date 2024-05-31 --symbols AAPL
   ```

8. Train model candidates from an explicit dataset version:

   ```bash
   quant-signal-pipeline train --dataset-version-id <dataset-version-id> --horizon 1 --horizon 5
   ```

9. Run a walk-forward backtest from an explicit model version:

   ```bash
   quant-signal-pipeline backtest --model-version-id <model-version-id> --top-n 1 --transaction-cost-bps 5 --slippage-bps 2
   ```

10. Generate SHAP explainability artifacts from an explicit model version:

   ```bash
   quant-signal-pipeline explain --model-version-id <model-version-id> --sample-size 8 --top-signals 3
   ```

## Validation Workflow

- `make lint`
- `make typecheck`
- `make test`
- `make validate`

## Pipeline CLI

- `quant-signal-pipeline ingest` is now available as the first thin orchestration command for scheduler-friendly market data ingestion
- `quant-signal-pipeline build-dataset` now maps directly to `FeaturePipeline` and prints a machine-readable dataset manifest summary with the persisted dataset version ID and artifact reference
- `quant-signal-pipeline train` now maps directly to `TrainingService` and prints machine-readable model metadata for each persisted trained candidate plus the champion model IDs for the requested horizons
- `quant-signal-pipeline backtest` now maps directly to `BacktestService` and prints a compact run summary with the persisted backtest run ID, artifact references, resolved execution assumptions, and key return statistics
- `quant-signal-pipeline explain` now maps directly to `ExplainabilityService` and prints a compact SHAP run summary with the persisted explainability run ID, artifact reference, realized sample size, and summary counts
- The commands delegate directly to `IngestionService`, `FeaturePipeline`, `TrainingService`, `BacktestService`, and `ExplainabilityService`, print JSON summaries on success, emit compact JSON error payloads on failure, and keep retry, dataset, training, backtest, and SHAP logic inside the service layer
- `--symbols` is optional; when omitted, the command uses `UNIVERSE_SYMBOLS` from settings
- `--feature-set-version` is optional for `build-dataset`; when omitted, the CLI defers to the service default
- `train` requires an explicit dataset version ID; it does not infer the latest dataset or chain dataset builds automatically
- `backtest` requires an explicit model version ID; optional `--top-n`, `--transaction-cost-bps`, and `--slippage-bps` override only the persisted service inputs and do not trigger training automatically
- `explain` requires an explicit model version ID; optional `--sample-size` and `--top-signals` map directly to the existing service inputs without exposing broader SHAP tuning
- The remaining pipeline subcommand for signal publication is intentionally deferred until a public refresh interface exists

## Current Status

The repository now includes the validated platform foundation, database schema, ingestion contract, persisted OHLCV workflow, feature engineering pipeline, versioned dataset artifacts, calibrated model training, cost-aware walk-forward backtesting with benchmark-relative analytics and richer regime context, SHAP explainability artifacts, persisted signal snapshots, and read-only FastAPI endpoints for health, signals, and model metadata.

## Implemented MVP Workflow

1. Ingest benchmark and universe OHLCV data through a provider abstraction.
2. Materialize reproducible feature datasets as versioned Parquet artifacts with registry metadata in PostgreSQL.
3. Train baseline `logistic_regression` and `hist_gradient_boosting` candidates per horizon, calibrate probabilities, and rank the champion with `PR-AUC`, `Brier score`, and `ROC-AUC`.
4. Persist ranked daily signal snapshots for champion models so the API stays read-only.
5. Run monthly walk-forward backtests with benchmark-relative analytics, cost-aware net returns, richer benchmark regime context, turnover-aware reporting, attribution-ready summaries, and regime-aware attribution slices.
6. Generate global and local SHAP summaries tied to a concrete model version and evaluation window.

## Provider Configuration

- `MARKET_DATA_PROVIDER` defaults to `yfinance`
- `MARKET_DATA_MAX_ATTEMPTS` defaults to `1`
- `MARKET_DATA_BACKOFF_SECONDS` defaults to `1.0`
- `MARKET_DATA_BACKOFF_MULTIPLIER` defaults to `2.0`
- Ingestion runs now persist nested request, provider, provider-fetch, and persistence metadata so partial or empty fetches are auditable without schema changes
- Ingestion failures are classified as transient or permanent at the provider edge, and transient fetch failures now retry deterministically before the run is finalized
- Runs persist a top-level `retry` block with attempt history, completed-after-retry state, and scheduled backoff values for retryable failed attempts

## Backtest Cost Assumptions

- `BACKTEST_TRANSACTION_COST_BPS` defaults to `0.0`
- `BACKTEST_SLIPPAGE_BPS` defaults to `0.0`
- Backtest artifacts now record daily `gross_return`, `transaction_cost`, `slippage_cost`, `net_return`, and `active_sleeves`
- `portfolio_return` remains available as an alias of `net_return` for backward compatibility

## Backtest Analytics

- Daily backtest artifacts include benchmark-relative fields such as `benchmark_return`, `active_return`, `gross_active_return`, and relative cumulative performance
- Benchmark regime context now includes the primary trend/volatility regime plus momentum and drawdown dimensions
- Persisted summaries include benchmark metrics, active-return metrics, and grouped performance slices for `trend_flag`, `volatility_flag`, `momentum_flag`, and `drawdown_bucket`
- Backtest runs also persist a companion detail artifact with composition-level rows for turnover and benchmark-relative contribution diagnostics
- Daily backtest artifacts now include `entries_count`, `exits_count`, `holdings_count`, `turnover`, and `turnover_cost`
- `summary_json` now includes top-level `attribution_metrics` and lifecycle-based `lifecycle_attribution` summaries keyed by `entry`, `held`, and `exit`
- `regime_summary_json` now includes daily average transaction/slippage/implementation drag fields for the primary regime keys, and `summary_json["attribution_dimension_summaries"]` carries the same daily-grain attribution view for each regime dimension
