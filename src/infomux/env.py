"""
Environment file loading for infomux.

Loads key/value pairs from a .env file into process environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from infomux.log import get_logger

logger = get_logger(__name__)

ENV_ENV_FILE = "INFOMUX_ENV_FILE"


def load_dotenv(path: Path | None = None) -> int:
    """
    Load environment variables from a dotenv-style file.

    By default this loads `.env` from the current working directory.
    Existing environment variables are never overridden.

    Args:
        path: Optional explicit path to env file.

    Returns:
        Number of variables loaded.
    """
    if path is None:
        explicit = os.environ.get(ENV_ENV_FILE)
        if explicit:
            path = Path(explicit).expanduser()
        else:
            path = Path.cwd() / ".env"

    if not path.exists():
        logger.debug("dotenv file not found: %s", path)
        return 0

    loaded = 0
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            logger.debug("ignoring invalid dotenv line %d", line_no)
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            logger.debug("ignoring empty dotenv key on line %d", line_no)
            continue

        # Remove optional matching quotes around the full value.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if key in os.environ:
            continue

        os.environ[key] = value
        loaded += 1

    logger.debug("loaded %d env vars from %s", loaded, path)
    return loaded
