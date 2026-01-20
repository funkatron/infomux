"""
Configuration management for infomux.

Handles environment variables and default paths for external tools.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from infomux.log import get_logger
from infomux.storage import get_data_dir

logger = get_logger(__name__)

# Environment variables
ENV_WHISPER_MODEL = "INFOMUX_WHISPER_MODEL"
ENV_FFMPEG_PATH = "INFOMUX_FFMPEG_PATH"
ENV_WHISPER_CLI_PATH = "INFOMUX_WHISPER_CLI_PATH"
ENV_TESSERACT_PATH = "INFOMUX_TESSERACT_PATH"


@dataclass
class ToolPaths:
    """
    Paths to external tools required by infomux.

    Attributes:
        ffmpeg: Path to ffmpeg binary.
        whisper_cli: Path to whisper-cli binary.
        whisper_model: Path to whisper model file.
        tesseract: Path to tesseract binary (for OCR).
    """

    ffmpeg: Path | None
    whisper_cli: Path | None
    whisper_model: Path | None
    tesseract: Path | None

    def validate(self) -> list[str]:
        """
        Validate that all required tools are available.

        Returns:
            List of error messages for missing tools (empty if all OK).
        """
        errors = []

        if not self.ffmpeg:
            errors.append("ffmpeg not found. Install via: brew install ffmpeg")
        elif not self.ffmpeg.exists():
            errors.append(f"ffmpeg not found at: {self.ffmpeg}")

        if not self.whisper_cli:
            errors.append(
                "whisper-cli not found. Install via: brew install whisper-cpp"
            )
        elif not self.whisper_cli.exists():
            errors.append(f"whisper-cli not found at: {self.whisper_cli}")

        if not self.whisper_model:
            errors.append(
                f"Whisper model not found. Set {ENV_WHISPER_MODEL} or place model at: "
                f"{get_default_whisper_model_path()}"
            )
        elif not self.whisper_model.exists():
            errors.append(f"Whisper model not found at: {self.whisper_model}")

        return errors


def get_default_whisper_model_path() -> Path:
    """
    Get the default path for the whisper model.

    Returns:
        Path to ~/.local/share/infomux/models/whisper/ggml-base.en.bin
    """
    return get_data_dir() / "models" / "whisper" / "ggml-base.en.bin"


def find_tool(name: str, env_var: str | None = None) -> Path | None:
    """
    Find a tool binary, checking env var first, then PATH.

    Args:
        name: Name of the tool binary.
        env_var: Optional environment variable to check first.

    Returns:
        Path to the tool, or None if not found.
    """
    # Check environment variable first
    if env_var:
        env_path = os.environ.get(env_var)
        if env_path:
            path = Path(env_path)
            if path.exists():
                logger.debug("found %s via %s: %s", name, env_var, path)
                return path
            else:
                logger.warning("%s set but file not found: %s", env_var, path)

    # Check PATH
    which_path = shutil.which(name)
    if which_path:
        path = Path(which_path)
        logger.debug("found %s in PATH: %s", name, path)
        return path

    logger.debug("%s not found", name)
    return None


def get_whisper_model_path() -> Path | None:
    """
    Get the path to the whisper model.

    Checks INFOMUX_WHISPER_MODEL env var first, then default location.

    Returns:
        Path to the model file, or None if not found.
    """
    # Check environment variable
    env_path = os.environ.get(ENV_WHISPER_MODEL)
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            logger.debug("found whisper model via %s: %s", ENV_WHISPER_MODEL, path)
            return path
        else:
            logger.warning("%s set but file not found: %s", ENV_WHISPER_MODEL, path)

    # Check default location
    default_path = get_default_whisper_model_path()
    if default_path.exists():
        logger.debug("found whisper model at default location: %s", default_path)
        return default_path

    logger.debug("whisper model not found")
    return None


def get_tool_paths() -> ToolPaths:
    """
    Discover all external tool paths.

    Returns:
        ToolPaths with discovered paths (some may be None if not found).
    """
    return ToolPaths(
        ffmpeg=find_tool("ffmpeg", ENV_FFMPEG_PATH),
        whisper_cli=find_tool("whisper-cli", ENV_WHISPER_CLI_PATH),
        whisper_model=get_whisper_model_path(),
        tesseract=find_tool("tesseract", ENV_TESSERACT_PATH),
    )
