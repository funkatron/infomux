"""
Job envelope management for infomux.

A job envelope is a JSON document that records all metadata about a pipeline run:
inputs, outputs, parameters, timing, and status. This provides full auditability
and enables resumption of interrupted runs.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(Enum):
    """Status of a pipeline job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class InputFile:
    """
    Metadata about an input file.

    Attributes:
        path: Absolute path to the input file.
        sha256: SHA-256 hash of the file contents for verification.
        size_bytes: File size in bytes.
    """

    path: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_path(cls, path: Path) -> InputFile:
        """
        Create InputFile metadata from a file path.

        Args:
            path: Path to the input file.

        Returns:
            InputFile with computed hash and size.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        # Compute SHA-256 hash
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return cls(
            path=str(path),
            sha256=sha256_hash.hexdigest(),
            size_bytes=path.stat().st_size,
        )


@dataclass
class StepRecord:
    """
    Record of a pipeline step execution.

    Attributes:
        name: Name of the step.
        status: Execution status.
        started_at: When the step started (ISO 8601).
        completed_at: When the step completed (ISO 8601), if finished.
        duration_seconds: How long the step took, if finished.
        error: Error message if the step failed.
        outputs: List of output artifact paths produced by this step.
        model_info: Model and generation parameters (for LLM steps).
    """

    name: str
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    outputs: list[str] = field(default_factory=list)
    model_info: dict[str, Any] | None = None


@dataclass
class JobEnvelope:
    """
    Complete job envelope for a pipeline run.

    This is the central record of a run, stored as job.json in the run directory.
    It provides full auditability and enables resumption of interrupted runs.

    Attributes:
        id: Unique identifier for this run.
        created_at: When the job was created (ISO 8601).
        updated_at: When the job was last updated (ISO 8601).
        status: Current status of the job.
        input: Metadata about the input file.
        steps: Records of each pipeline step.
        artifacts: List of all output artifact paths.
        config: Configuration used for this run.
        error: Top-level error message if the job failed.
    """

    id: str
    created_at: str
    updated_at: str
    status: str
    input: InputFile | None = None
    steps: list[StepRecord] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def create(cls, input_file: InputFile | None = None) -> JobEnvelope:
        """
        Create a new job envelope.

        Args:
            input_file: Optional input file metadata.

        Returns:
            New JobEnvelope with a unique ID and timestamps.
        """
        now = datetime.now(UTC).isoformat()
        run_id = generate_run_id()

        return cls(
            id=run_id,
            created_at=now,
            updated_at=now,
            status=JobStatus.PENDING.value,
            input=input_file,
        )

    def update_status(self, status: JobStatus, error: str | None = None) -> None:
        """
        Update the job status and timestamp.

        Args:
            status: New status.
            error: Optional error message (for FAILED status).
        """
        self.status = status.value
        self.updated_at = datetime.now(UTC).isoformat()
        if error:
            self.error = error

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the envelope to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the job envelope.
        """
        data = asdict(self)
        # Convert InputFile to dict if present
        if self.input:
            data["input"] = asdict(self.input)
        return data

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the envelope to JSON.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobEnvelope:
        """
        Create a JobEnvelope from a dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            JobEnvelope instance.
        """
        # Handle nested InputFile
        input_data = data.get("input")
        if input_data:
            data["input"] = InputFile(**input_data)

        # Handle nested StepRecords
        steps_data = data.get("steps", [])
        data["steps"] = [StepRecord(**s) for s in steps_data]

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> JobEnvelope:
        """
        Create a JobEnvelope from a JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            JobEnvelope instance.
        """
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path) -> JobEnvelope:
        """
        Load a JobEnvelope from a file.

        Args:
            path: Path to the job.json file.

        Returns:
            JobEnvelope instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(path) as f:
            return cls.from_json(f.read())

    def save(self, path: Path) -> None:
        """
        Save the JobEnvelope to a file.

        Args:
            path: Path to write the job.json file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


def generate_run_id() -> str:
    """
    Generate a unique run ID.

    Format: run-YYYYMMDD-HHMMSS-XXXXXX
    where XXXXXX is a short random suffix for uniqueness.

    Returns:
        Unique run identifier.
    """
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"run-{timestamp}-{suffix}"
