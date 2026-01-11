"""
Tests for the resume command.
"""

from __future__ import annotations

from pathlib import Path

from infomux.job import InputFile, JobEnvelope, StepRecord
from infomux.pipeline import get_resumable_steps


class TestGetResumableSteps:
    """Tests for get_resumable_steps."""

    def test_no_steps_completed(self, tmp_path: Path) -> None:
        """All steps are resumable when none completed."""
        job = _create_test_job(tmp_path, steps=[])
        job.config["pipeline"] = "transcribe"

        steps = get_resumable_steps(job)

        assert steps == ["extract_audio", "transcribe"]

    def test_first_step_completed(self, tmp_path: Path) -> None:
        """Only incomplete steps are resumable."""
        job = _create_test_job(
            tmp_path,
            steps=[StepRecord(name="extract_audio", status="completed")],
        )
        job.config["pipeline"] = "transcribe"

        steps = get_resumable_steps(job)

        assert steps == ["transcribe"]

    def test_all_steps_completed(self, tmp_path: Path) -> None:
        """No steps resumable when all completed."""
        job = _create_test_job(
            tmp_path,
            steps=[
                StepRecord(name="extract_audio", status="completed"),
                StepRecord(name="transcribe", status="completed"),
            ],
        )
        job.config["pipeline"] = "transcribe"

        steps = get_resumable_steps(job)

        assert steps == []

    def test_from_step_specified(self, tmp_path: Path) -> None:
        """--from-step resumes from that step onwards."""
        job = _create_test_job(
            tmp_path,
            steps=[
                StepRecord(name="extract_audio", status="completed"),
                StepRecord(name="transcribe", status="completed"),
            ],
        )
        job.config["pipeline"] = "transcribe"

        steps = get_resumable_steps(job, from_step="transcribe")

        assert steps == ["transcribe"]

    def test_from_step_first(self, tmp_path: Path) -> None:
        """--from-step extract_audio re-runs entire pipeline."""
        job = _create_test_job(
            tmp_path,
            steps=[
                StepRecord(name="extract_audio", status="completed"),
                StepRecord(name="transcribe", status="completed"),
            ],
        )
        job.config["pipeline"] = "transcribe"

        steps = get_resumable_steps(job, from_step="extract_audio")

        assert steps == ["extract_audio", "transcribe"]

    def test_from_step_unknown(self, tmp_path: Path) -> None:
        """Unknown --from-step returns empty list."""
        job = _create_test_job(tmp_path, steps=[])
        job.config["pipeline"] = "transcribe"

        steps = get_resumable_steps(job, from_step="nonexistent")

        assert steps == []

    def test_failed_step_is_resumable(self, tmp_path: Path) -> None:
        """Failed steps are included in resumable."""
        job = _create_test_job(
            tmp_path,
            steps=[
                StepRecord(name="extract_audio", status="completed"),
                StepRecord(name="transcribe", status="failed", error="test error"),
            ],
        )
        job.config["pipeline"] = "transcribe"

        steps = get_resumable_steps(job)

        assert steps == ["transcribe"]


def _create_test_job(
    tmp_path: Path,
    steps: list[StepRecord] | None = None,
) -> JobEnvelope:
    """Create a test job envelope."""
    test_file = tmp_path / "test.mp4"
    test_file.write_bytes(b"test content")

    input_file = InputFile.from_path(test_file)
    job = JobEnvelope.create(input_file=input_file)
    job.steps = steps or []
    return job
