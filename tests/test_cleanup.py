"""
Tests for the cleanup command.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime, timedelta

import pytest

from infomux.commands.cleanup import execute, parse_time_spec
from infomux.job import JobEnvelope, JobStatus
from infomux.storage import save_job


def _save_job_with_age(days_old: int, status: JobStatus) -> JobEnvelope:
    job = JobEnvelope.create()
    created = datetime.now(UTC) - timedelta(days=days_old)
    job.created_at = created.isoformat()
    job.updated_at = created.isoformat()
    job.status = status.value
    save_job(job)
    return job


def test_parse_time_spec_rejects_non_positive_values() -> None:
    """Time specs must be strictly positive."""
    with pytest.raises(ValueError, match="must be > 0"):
        parse_time_spec("0d")
    with pytest.raises(ValueError, match="must be > 0"):
        parse_time_spec("-1d")


def test_cleanup_rejects_negative_older_than() -> None:
    """Negative time specs are rejected by execute()."""
    args = Namespace(
        dry_run=True,
        force=False,
        orphaned=False,
        status=None,
        older_than="-1d",
        min_age=None,
    )
    assert execute(args) == 1


def test_cleanup_status_and_older_than_are_conjunctive(
    tmp_path, monkeypatch, capsys
) -> None:
    """Combining status and age filters should narrow matches."""
    monkeypatch.setenv("INFOMUX_DATA_DIR", str(tmp_path))

    old_failed = _save_job_with_age(days_old=10, status=JobStatus.FAILED)
    new_failed = _save_job_with_age(days_old=1, status=JobStatus.FAILED)
    _save_job_with_age(days_old=20, status=JobStatus.COMPLETED)

    args = Namespace(
        dry_run=True,
        force=False,
        orphaned=False,
        status="failed",
        older_than="7d",
        min_age=None,
    )
    assert execute(args) == 0
    out = capsys.readouterr().out

    assert old_failed.id in out
    assert new_failed.id not in out
