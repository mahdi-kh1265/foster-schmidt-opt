"""Logging configuration for foster_eom.

Provides a consistent logging setup across the package.  The GUI and CLI
configure the root handler; library code uses named loggers.
"""

from __future__ import annotations

import logging
import sys
from typing import IO

PACKAGE_LOGGER_NAME = "foster_eom"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the package namespace.

    Parameters
    ----------
    name : str
        Module or component name (e.g. ``"foster.seed_solver"``).

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


def configure_logging(
    level: int = logging.INFO,
    stream: IO[str] | None = None,
) -> None:
    """Configure the package root logger.

    Parameters
    ----------
    level : int
        Logging level (default ``INFO``).
    stream : file-like or None
        Output stream (default ``sys.stderr``).
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if logger.handlers:
        # Already configured — avoid duplicate handlers
        return

    handler = logging.StreamHandler(stream or sys.stderr)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
