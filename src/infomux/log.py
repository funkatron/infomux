"""
Logging configuration for infomux.

All logs are written to stderr to keep stdout clean for machine-readable output.
"""

from __future__ import annotations

import logging
import os
import sys

# Default log format: timestamp, level, logger name, message
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Environment variable for log level override
ENV_LOG_LEVEL = "INFOMUX_LOG_LEVEL"


def configure_logging(
    level: str | None = None,
    format_string: str = LOG_FORMAT,
    date_format: str = LOG_DATE_FORMAT,
) -> None:
    """
    Configure logging for infomux.

    All output goes to stderr. The log level can be set via:
    1. The `level` parameter
    2. The INFOMUX_LOG_LEVEL environment variable
    3. Default: INFO

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). If None, uses env or default.
        format_string: Log message format.
        date_format: Timestamp format.
    """
    # Determine log level
    if level is None:
        level = os.environ.get(ENV_LOG_LEVEL, "INFO")

    level = level.upper()

    # Map level string to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = level_map.get(level, logging.INFO)

    # Configure root logger for infomux
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(format_string, date_format))

    # Configure the infomux logger hierarchy
    infomux_logger = logging.getLogger("infomux")
    infomux_logger.setLevel(log_level)
    infomux_logger.handlers.clear()
    infomux_logger.addHandler(handler)
    infomux_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance configured for infomux.
    """
    # Ensure the name is under the infomux hierarchy
    if not name.startswith("infomux"):
        name = f"infomux.{name}"
    return logging.getLogger(name)
