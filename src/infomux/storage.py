"""
Storage utilities for infomux.

Handles the file-backed storage of runs and artifacts under the data directory.
Default location: ~/.local/share/infomux/runs/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from infomux.log import get_logger

if TYPE_CHECKING:
    from infomux.job import JobEnvelope

logger = get_logger(__name__)

# Environment variable for data directory override
ENV_DATA_DIR = "INFOMUX_DATA_DIR"

# Default data directory following XDG Base Directory spec
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "infomux"


def get_data_dir() -> Path:
    """
    Get the base data directory for infomux.

    Respects the INFOMUX_DATA_DIR environment variable, falling back
    to ~/.local/share/infomux if not set.

    Returns:
        Path to the data directory.
    """
    env_dir = os.environ.get(ENV_DATA_DIR)
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return DEFAULT_DATA_DIR


def get_runs_dir() -> Path:
    """
    Get the directory where runs are stored.

    Returns:
        Path to the runs directory.
    """
    return get_data_dir() / "runs"


def get_run_dir(run_id: str) -> Path:
    """
    Get the directory for a specific run.

    Args:
        run_id: The unique run identifier.

    Returns:
        Path to the run directory.
    """
    return get_runs_dir() / run_id


def get_job_path(run_id: str) -> Path:
    """
    Get the path to the job.json file for a run.

    Args:
        run_id: The unique run identifier.

    Returns:
        Path to the job.json file.
    """
    return get_run_dir(run_id) / "job.json"


def create_run_dir(run_id: str) -> Path:
    """
    Create the directory for a new run.

    Args:
        run_id: The unique run identifier.

    Returns:
        Path to the created run directory.
    """
    run_dir = get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("created run directory: %s", run_dir)
    return run_dir


def run_exists(run_id: str) -> bool:
    """
    Check if a run exists.

    Args:
        run_id: The unique run identifier.

    Returns:
        True if the run directory and job.json exist.
    """
    job_path = get_job_path(run_id)
    return job_path.exists()


def list_runs() -> list[str]:
    """
    List all run IDs in the runs directory.

    Returns:
        List of run IDs, sorted by name (most recent first due to timestamp format).
    """
    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        return []

    run_ids = []
    for entry in runs_dir.iterdir():
        if entry.is_dir() and (entry / "job.json").exists():
            run_ids.append(entry.name)

    # Sort descending (most recent first)
    return sorted(run_ids, reverse=True)


def save_job(job: JobEnvelope) -> Path:
    """
    Save a job envelope to its run directory.

    Creates the run directory if it doesn't exist.

    Args:
        job: The job envelope to save.

    Returns:
        Path to the saved job.json file.
    """
    run_dir = create_run_dir(job.id)
    job_path = run_dir / "job.json"
    job.save(job_path)
    logger.debug("saved job envelope: %s", job_path)
    return job_path


def load_job(run_id: str) -> JobEnvelope:
    """
    Load a job envelope from a run directory.

    Args:
        run_id: The unique run identifier.

    Returns:
        The loaded job envelope.

    Raises:
        FileNotFoundError: If the run does not exist.
    """
    from infomux.job import JobEnvelope

    job_path = get_job_path(run_id)
    if not job_path.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")

    logger.debug("loading job envelope: %s", job_path)
    return JobEnvelope.load(job_path)
