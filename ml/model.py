"""
Chronos-2 model loading (Nishit's implementation, made lazy and cached).

Nishit's original predict.py loaded the model at import time, which made
every import slow and crashed any module that merely imported it when the
model could not be downloaded. The model is now loaded once, on first use.
"""

from __future__ import annotations

import logging
import threading

import config

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Chronos-2 could not be imported or loaded."""


_pipeline = None
_lock = threading.Lock()


def load_model(model_name: str | None = None, device: str | None = None):
    """
    Load Chronos2Pipeline exactly as Nishit did:
        Chronos2Pipeline.from_pretrained("amazon/chronos-2",
                                         device_map="cpu", dtype=torch.float32)
    Raises ModelUnavailableError with a clear message on failure.
    """
    name = model_name or config.CHRONOS_MODEL_NAME
    dev = device or config.CHRONOS_DEVICE
    try:
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ModelUnavailableError(
            "chronos-forecasting / torch not installed. Run: pip install -r requirements.txt"
        ) from exc

    logger.info("Loading %s on %s ...", name, dev)
    try:
        pipeline = Chronos2Pipeline.from_pretrained(name, device_map=dev, dtype=torch.float32)
    except Exception as exc:  # network, missing weights, bad device, ...
        raise ModelUnavailableError(f"Could not load {name}: {exc}") from exc
    logger.info("%s loaded", name)
    return pipeline


def get_pipeline():
    """Return the shared Chronos-2 pipeline, loading it on first call."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = load_model()
    return _pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_pipeline()
    print("Chronos-2 is ready for forecasting.")
