"""
Common storage API for infomux.

Defines the interface and utilities shared by all storage steps.
Storage steps read run artifacts and persist them to various backends.

Design principles:
- Files in run directory are the source of truth
- Stores are derived indexes (can be rebuilt)
- Common RunData structure for all stores
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infomux.log import get_logger

logger = get_logger(__name__)


@dataclass
class Segment:
    """A transcript segment with timing."""

    start_ms: int
    end_ms: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        return cls(
            start_ms=data.get("start_ms", 0),
            end_ms=data.get("end_ms", 0),
            text=data.get("text", ""),
        )


@dataclass
class RunData:
    """
    Unified data structure for a run.

    Collected from various artifacts in the run directory.
    Used by all storage backends for consistent data access.
    """

    # Metadata
    run_id: str
    created_at: str
    input_path: str | None = None
    input_hash: str | None = None
    pipeline: str | None = None
    status: str | None = None
    duration_seconds: float | None = None

    # Content
    transcript_text: str | None = None
    transcript_segments: list[Segment] = field(default_factory=list)
    summary: str | None = None
    summary_model: str | None = None

    # Raw data
    job_json: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "input_path": self.input_path,
            "input_hash": self.input_hash,
            "pipeline": self.pipeline,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "transcript_text": self.transcript_text,
            "transcript_segments": [s.to_dict() for s in self.transcript_segments],
            "summary": self.summary,
            "summary_model": self.summary_model,
        }

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> RunData | None:
        """
        Load RunData from a run directory.

        Args:
            run_dir: Path to the run directory.

        Returns:
            RunData if job.json exists, None otherwise.
        """
        job_path = run_dir / "job.json"
        if not job_path.exists():
            logger.warning("job.json not found in %s", run_dir)
            return None

        with open(job_path) as f:
            job = json.load(f)

        run_id = job.get("id")
        if not run_id:
            return None

        # Build RunData
        data = cls(
            run_id=run_id,
            created_at=job.get("created_at", ""),
            input_path=job.get("input", {}).get("path"),
            input_hash=job.get("input", {}).get("sha256"),
            pipeline=job.get("config", {}).get("pipeline"),
            status=job.get("status"),
            job_json=job,
        )

        # Calculate duration from steps
        total_duration = 0.0
        for step in job.get("steps", []):
            if step.get("duration_seconds"):
                total_duration += step["duration_seconds"]
            # Get summary model info
            if step.get("name") == "summarize" and step.get("model_info"):
                data.summary_model = step["model_info"].get("model", {}).get("name")
        data.duration_seconds = total_duration

        # Load transcript text
        transcript_path = run_dir / "transcript.txt"
        if transcript_path.exists():
            data.transcript_text = transcript_path.read_text()

        # Load segments from JSON
        json_path = run_dir / "transcript.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    transcript_data = json.load(f)

                for segment in transcript_data.get("transcription", []):
                    offsets = segment.get("offsets", {})
                    data.transcript_segments.append(
                        Segment(
                            start_ms=offsets.get("from", 0),
                            end_ms=offsets.get("to", 0),
                            text=segment.get("text", "").strip(),
                        )
                    )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to parse transcript.json: %s", e)

        # Load summary
        summary_path = run_dir / "summary.md"
        if summary_path.exists():
            data.summary = summary_path.read_text()

        return data


def format_duration(seconds: float | None) -> str:
    """Format duration as human-readable string."""
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def format_timestamp(ms: int) -> str:
    """Format milliseconds as HH:MM:SS.mmm."""
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    millis = ms % 1000
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"
