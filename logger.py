"""Centralized logging for parallel-encoder."""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "parallel-encoder"

_VERBOSITY_MAP = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def setup_logging(
    verbosity: int = 0,
    log_file: str | None = None,
) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        verbosity: 0 = WARNING, 1 = INFO, 2 = DEBUG.
        log_file: Optional path to a log file. When set, all messages
            at DEBUG level are written to this file regardless of
            the console verbosity.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    # Clear any existing handlers (avoids duplicates on repeated calls)
    logger.handlers.clear()

    level = _VERBOSITY_MAP.get(verbosity, logging.DEBUG)
    logger.setLevel(min(level, logging.DEBUG) if log_file else level)

    # Console handler (respects verbosity)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console)

    # File handler (always DEBUG when enabled)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(fh)

    return logger
