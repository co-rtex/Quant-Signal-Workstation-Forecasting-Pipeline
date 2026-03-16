"""API dependency helpers."""

from quant_signal.serving.service import SignalService


def get_signal_service() -> SignalService:
    """Return the read-only signal service."""

    return SignalService()
