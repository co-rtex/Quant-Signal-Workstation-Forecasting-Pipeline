"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest

from quant_signal.core.config import get_settings
from quant_signal.storage.db import clear_engine_cache


@pytest.fixture(autouse=True)
def clear_cached_configuration() -> Iterator[None]:
    """Clear cached settings and engine state between tests."""

    get_settings.cache_clear()
    clear_engine_cache()
    yield
    get_settings.cache_clear()
    clear_engine_cache()
