"""Logging configuration."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path

from src.utils.constants import OUTPUT_DIR

LOG_FILE = OUTPUT_DIR / "logs.txt"


def get_logger(name: str) -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


@contextmanager
def timed_step(logger: logging.Logger, label: str):
    start = time.perf_counter()
    logger.info("Starting: %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Finished: %s (%.2fs)", label, elapsed)
