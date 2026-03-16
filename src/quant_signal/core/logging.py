"""Logging helpers."""

import logging


def configure_logging(level: str) -> None:
    """Configure the root logger once."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
