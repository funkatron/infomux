"""
Tests for the job envelope module.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from infomux.job import (
    InputFile,
    JobEnvelope,
    JobStatus,
    StepRecord,
    generate_run_id,
)


class TestGenerateRunId:
    """Tests for run ID generation."""

    def test_format(self) -> None:
        """Run ID follows expected format."""
        run_id = generate_run_id()
        assert run_id.startswith("run-")
        parts = run_id.split("-")
        assert len(parts) == 4  # run, date, time, suffix

    def test_uniqueness(self) -> None:
        """Generated run IDs are unique."""
        ids = [generate_run_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestInputFile:
    """Tests for InputFile metadata."""

    def test_from_path(self, tmp_path: Path) -> None:
        """InputFile correctly reads file metadata."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        input_file = InputFile.from_path(test_file)

        assert input_file.path == str(test_file)
        assert input_file.size_bytes == 11
        assert len(input_file.sha256) == 64  # SHA-256 hex length

    def test_from_path_not_found(self, tmp_path: Path) -> None:
        """InputFile raises for non-existent files."""
        with pytest.raises(FileNotFoundError):
            InputFile.from_path(tmp_path / "nonexistent.txt")


class TestJobEnvelope:
    """Tests for JobEnvelope."""

    def test_create(self) -> None:
        """JobEnvelope.create produces valid envelope."""
        job = JobEnvelope.create()

        assert job.id.startswith("run-")
        assert job.status == JobStatus.PENDING.value
        assert job.created_at is not None
        assert job.updated_at is not None
        assert job.steps == []
        assert job.artifacts == []

    def test_create_with_input(self, tmp_path: Path) -> None:
        """JobEnvelope.create includes input file metadata."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video content")
        input_file = InputFile.from_path(test_file)

        job = JobEnvelope.create(input_file=input_file)

        assert job.input is not None
        assert job.input.path == str(test_file)

    def test_update_status(self) -> None:
        """update_status changes status and updates timestamp."""
        job = JobEnvelope.create()
        original_updated = job.updated_at

        job.update_status(JobStatus.RUNNING)

        assert job.status == JobStatus.RUNNING.value
        assert job.updated_at >= original_updated

    def test_to_json_and_back(self, tmp_path: Path) -> None:
        """JobEnvelope round-trips through JSON."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video")
        input_file = InputFile.from_path(test_file)

        original = JobEnvelope.create(input_file=input_file)
        original.steps.append(StepRecord(name="test_step", status="completed"))
        original.artifacts.append("output.txt")

        # Round trip
        json_str = original.to_json()
        restored = JobEnvelope.from_json(json_str)

        assert restored.id == original.id
        assert restored.status == original.status
        assert restored.input.sha256 == original.input.sha256
        assert len(restored.steps) == 1
        assert restored.steps[0].name == "test_step"
        assert restored.artifacts == ["output.txt"]

    def test_save_and_load(self, tmp_path: Path) -> None:
        """JobEnvelope saves and loads from file."""
        job = JobEnvelope.create()
        job_path = tmp_path / "job.json"

        job.save(job_path)
        loaded = JobEnvelope.load(job_path)

        assert loaded.id == job.id
        assert loaded.status == job.status
