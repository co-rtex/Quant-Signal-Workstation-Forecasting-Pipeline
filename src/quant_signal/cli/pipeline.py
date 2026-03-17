"""Pipeline CLI entrypoints."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, TextIO

from quant_signal.core.config import Settings
from quant_signal.core.logging import configure_logging
from quant_signal.features.pipeline import FeaturePipeline
from quant_signal.ingestion.service import IngestionService
from quant_signal.storage.models import DatasetVersion, IngestionRun

logger = logging.getLogger(__name__)

IngestionServiceFactory = Callable[[Settings], IngestionService]
FeaturePipelineFactory = Callable[[Settings], FeaturePipeline]


def _parse_date(value: str) -> date:
    """Parse an ISO-8601 date argument."""

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    """Normalize user-provided symbol input into unique uppercase tickers."""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = raw_symbol.strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return normalized


def _build_ingestion_service(settings: Settings) -> IngestionService:
    """Build the ingestion service for CLI execution."""

    return IngestionService(settings=settings)


def _build_feature_pipeline(settings: Settings) -> FeaturePipeline:
    """Build the feature pipeline for CLI execution."""

    return FeaturePipeline(settings=settings)


@dataclass(frozen=True)
class ServiceFactories:
    """Factory bundle used by command handlers."""

    ingestion: IngestionServiceFactory = _build_ingestion_service
    features: FeaturePipelineFactory = _build_feature_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the shared pipeline command parser."""

    parser = argparse.ArgumentParser(
        prog="quant-signal-pipeline",
        description="Thin pipeline orchestration commands for quant-signal services.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Fetch and persist normalized daily market data.",
    )
    ingest_parser.add_argument(
        "--start-date",
        required=True,
        type=_parse_date,
        help="Inclusive ISO start date, for example 2024-01-01.",
    )
    ingest_parser.add_argument(
        "--end-date",
        required=True,
        type=_parse_date,
        help="Inclusive ISO end date, for example 2024-01-31.",
    )
    ingest_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Optional space-delimited tickers. Defaults to UNIVERSE_SYMBOLS when omitted.",
    )
    ingest_parser.set_defaults(handler=_handle_ingest)

    build_dataset_parser = subparsers.add_parser(
        "build-dataset",
        help="Materialize and register a versioned feature dataset.",
    )
    build_dataset_parser.add_argument(
        "--as-of-date",
        required=True,
        type=_parse_date,
        help="Inclusive ISO dataset cutoff date, for example 2024-05-31.",
    )
    build_dataset_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Optional space-delimited tickers. Defaults to UNIVERSE_SYMBOLS when omitted.",
    )
    build_dataset_parser.add_argument(
        "--feature-set-version",
        help="Optional feature set version. Defaults to the service default when omitted.",
    )
    build_dataset_parser.set_defaults(handler=_handle_build_dataset)

    return parser


def _handle_ingest(
    args: argparse.Namespace,
    settings: Settings,
    service_factories: ServiceFactories,
) -> dict[str, object]:
    """Execute the ingestion command and return a machine-readable summary."""

    requested_symbols = _normalize_symbols(args.symbols or settings.universe_symbols)
    run = service_factories.ingestion(settings).ingest_daily_bars(
        requested_symbols,
        args.start_date,
        args.end_date,
    )
    return _summarize_ingestion_run(run)


def _handle_build_dataset(
    args: argparse.Namespace,
    settings: Settings,
    service_factories: ServiceFactories,
) -> dict[str, object]:
    """Execute the dataset build command and return a machine-readable summary."""

    requested_symbols = _normalize_symbols(args.symbols) if args.symbols else None
    pipeline = service_factories.features(settings)

    if args.feature_set_version is None:
        dataset = pipeline.build_dataset(args.as_of_date, requested_symbols)
    else:
        dataset = pipeline.build_dataset(
            args.as_of_date,
            requested_symbols,
            args.feature_set_version,
        )

    return _summarize_dataset_version(dataset)


def _summarize_ingestion_run(run: IngestionRun) -> dict[str, object]:
    """Build a stable JSON summary for ingestion command output."""

    metadata = run.metadata_json
    request_metadata = _metadata_section(metadata, "request")
    retry_metadata = _metadata_section(metadata, "retry")
    return {
        "command": "ingest",
        "run_id": run.id,
        "status": run.status,
        "provider": run.provider,
        "records_written": run.records_written,
        "retry_attempt_count": int(retry_metadata.get("attempt_count", 0)),
        "completed_after_retry": bool(retry_metadata.get("completed_after_retry", False)),
        "requested_symbols": request_metadata.get("requested_symbols", run.requested_symbols),
        "fetch_symbols": request_metadata.get("fetch_symbols", []),
        "benchmark_symbol": request_metadata.get("benchmark_symbol"),
        "start_date": run.start_date.isoformat(),
        "end_date": run.end_date.isoformat(),
    }


def _summarize_dataset_version(dataset: DatasetVersion) -> dict[str, object]:
    """Build a stable JSON summary for dataset command output."""

    metadata = dataset.metadata_json
    date_range = _metadata_section(metadata, "date_range")
    return {
        "command": "build-dataset",
        "status": "completed",
        "dataset_version_id": dataset.id,
        "as_of_date": dataset.as_of_date.isoformat(),
        "feature_set_version": dataset.feature_set_version,
        "row_count": dataset.row_count,
        "symbols": dataset.symbols,
        "horizons": dataset.horizons,
        "artifact_path": dataset.artifact_path,
        "artifact_hash": dataset.artifact_hash,
        "benchmark_symbol": metadata.get("benchmark_symbol"),
        "date_range": date_range,
    }


def _metadata_section(metadata: dict[str, object], key: str) -> dict[str, Any]:
    """Return a nested metadata section when present and dictionary-shaped."""

    section = metadata.get(key)
    if isinstance(section, dict):
        return section
    return {}


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    service_factories: ServiceFactories | None = None,
) -> int:
    """Execute the pipeline CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    resolved_settings = settings or Settings()
    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    factories = service_factories or ServiceFactories()

    configure_logging(resolved_settings.log_level)

    try:
        handler = args.handler
        payload = handler(args, resolved_settings, factories)
    except Exception as exc:
        logger.exception("Pipeline command %s failed", args.command)
        json.dump(
            {
                "command": args.command,
                "status": "failed",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
            error_output,
            sort_keys=True,
        )
        error_output.write("\n")
        return 1

    json.dump(payload, output, sort_keys=True)
    output.write("\n")
    return 0


def run() -> None:
    """Entrypoint for the packaged pipeline script."""

    raise SystemExit(main())
