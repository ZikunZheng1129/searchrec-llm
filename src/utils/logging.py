"""Logging helpers."""

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger that is safe to request repeatedly."""
    logger = logging.getLogger(name)
    log_level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(log_level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(log_level)

    return logger
