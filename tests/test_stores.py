"""
Tests for storage steps.

Principle: Test our code logic, assume external libraries work.
- Mock sqlite3, boto3, psycopg2, subprocess
- Test data transformation and error handling
- Don't test that SQLite can store data (it can)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.steps.storage import RunData, Segment, format_duration, format_timestamp


# =============================================================================
# Test RunData (common storage API)
# =============================================================================


class TestSegment:
    """Test Segment dataclass."""

    def test_to_dict(self):
        seg = Segment(start_ms=1000, end_ms=2000, text="hello")
        assert seg.to_dict() == {
            "start_ms": 1000,
            "end_ms": 2000,
            "text": "hello",
        }

    def test_from_dict(self):
        data = {"start_ms": 500, "end_ms": 1500, "text": "world"}
        seg = Segment.from_dict(data)
        assert seg.start_ms == 500
        assert seg.end_ms == 1500
        assert seg.text == "world"


class TestRunData:
    """Test RunData dataclass."""

    def test_to_dict(self):
        data = RunData(
            run_id="run-123",
            created_at="2026-01-10T12:00:00",
            input_path="/path/to/file.mp4",
            transcript_text="Hello world",
        )
        result = data.to_dict()
        assert result["run_id"] == "run-123"
        assert result["input_path"] == "/path/to/file.mp4"
        assert result["transcript_text"] == "Hello world"

    def test_from_run_dir_no_job_json(self, tmp_path):
        """Returns None if job.json doesn't exist."""
        result = RunData.from_run_dir(tmp_path)
        assert result is None

    def test_from_run_dir_with_job_json(self, tmp_path):
        """Loads data from job.json."""
        job = {
            "id": "run-test-123",
            "created_at": "2026-01-10T12:00:00",
            "status": "completed",
            "input": {"path": "/test/file.mp4", "sha256": "abc123"},
            "config": {"pipeline": "transcribe"},
            "steps": [],
        }
        (tmp_path / "job.json").write_text(json.dumps(job))

        result = RunData.from_run_dir(tmp_path)
        assert result is not None
        assert result.run_id == "run-test-123"
        assert result.status == "completed"
        assert result.pipeline == "transcribe"

    def test_from_run_dir_loads_transcript(self, tmp_path):
        """Loads transcript.txt if present."""
        job = {"id": "run-123", "created_at": "2026-01-10", "steps": []}
        (tmp_path / "job.json").write_text(json.dumps(job))
        (tmp_path / "transcript.txt").write_text("Hello, this is a test.")

        result = RunData.from_run_dir(tmp_path)
        assert result.transcript_text == "Hello, this is a test."

    def test_from_run_dir_loads_summary(self, tmp_path):
        """Loads summary.md if present."""
        job = {"id": "run-123", "created_at": "2026-01-10", "steps": []}
        (tmp_path / "job.json").write_text(json.dumps(job))
        (tmp_path / "summary.md").write_text("This is the summary.")

        result = RunData.from_run_dir(tmp_path)
        assert result.summary == "This is the summary."

    def test_from_run_dir_loads_segments(self, tmp_path):
        """Loads segments from transcript.json."""
        job = {"id": "run-123", "created_at": "2026-01-10", "steps": []}
        transcript_json = {
            "transcription": [
                {"offsets": {"from": 0, "to": 1000}, "text": "Hello"},
                {"offsets": {"from": 1000, "to": 2000}, "text": "World"},
            ]
        }
        (tmp_path / "job.json").write_text(json.dumps(job))
        (tmp_path / "transcript.json").write_text(json.dumps(transcript_json))

        result = RunData.from_run_dir(tmp_path)
        assert len(result.transcript_segments) == 2
        assert result.transcript_segments[0].text == "Hello"
        assert result.transcript_segments[1].start_ms == 1000


class TestFormatters:
    """Test formatting helper functions."""

    def test_format_duration_seconds(self):
        assert format_duration(30.5) == "30.5s"

    def test_format_duration_minutes(self):
        assert format_duration(90) == "1m 30s"

    def test_format_duration_none(self):
        assert format_duration(None) == "unknown"

    def test_format_timestamp_minutes(self):
        assert format_timestamp(65000) == "01:05.000"

    def test_format_timestamp_hours(self):
        assert format_timestamp(3661000) == "01:01:01.000"


# =============================================================================
# Test store_json
# =============================================================================


class TestStoreJson:
    """Test store_json step."""

    def test_creates_report_json(self, tmp_path):
        """Creates report.json with run data."""
        from infomux.steps.store_json import StoreJsonStep

        # Setup run directory
        job = {"id": "run-json-test", "created_at": "2026-01-10", "steps": []}
        (tmp_path / "job.json").write_text(json.dumps(job))
        (tmp_path / "transcript.txt").write_text("Test transcript")

        step = StoreJsonStep()
        outputs = step.execute(tmp_path / "audio.wav", tmp_path)

        assert len(outputs) == 1
        assert outputs[0].name == "report.json"
        assert outputs[0].exists()

        # Verify content structure
        content = json.loads(outputs[0].read_text())
        assert "version" in content
        assert "exported_at" in content
        assert content["run"]["run_id"] == "run-json-test"

    def test_run_function_success(self, tmp_path):
        """run() returns successful StepResult."""
        from infomux.steps.store_json import run

        job = {"id": "run-123", "created_at": "2026-01-10", "steps": []}
        (tmp_path / "job.json").write_text(json.dumps(job))

        result = run(tmp_path / "audio.wav", tmp_path)
        assert result.success
        assert result.name == "store_json"

    def test_run_function_no_data(self, tmp_path):
        """run() fails gracefully when no job.json."""
        from infomux.steps.store_json import run

        result = run(tmp_path / "audio.wav", tmp_path)
        assert not result.success
        assert "No run data" in result.error


# =============================================================================
# Test store_markdown
# =============================================================================


class TestStoreMarkdown:
    """Test store_markdown step."""

    def test_creates_report_md(self, tmp_path):
        """Creates report.md with formatted content."""
        from infomux.steps.store_markdown import StoreMarkdownStep

        job = {
            "id": "run-md-test",
            "created_at": "2026-01-10T12:00:00",
            "steps": [],
            "input": {"path": "/test/file.mp4"},
        }
        (tmp_path / "job.json").write_text(json.dumps(job))
        (tmp_path / "transcript.txt").write_text("Test transcript content")

        step = StoreMarkdownStep()
        outputs = step.execute(tmp_path / "audio.wav", tmp_path)

        assert len(outputs) == 1
        assert outputs[0].name == "report.md"

        content = outputs[0].read_text()
        assert "# Transcription Report" in content
        assert "run-md-test" in content
        assert "Test transcript content" in content

    def test_includes_summary_when_present(self, tmp_path):
        """Includes summary section when summary.md exists."""
        from infomux.steps.store_markdown import StoreMarkdownStep

        job = {"id": "run-123", "created_at": "2026-01-10", "steps": []}
        (tmp_path / "job.json").write_text(json.dumps(job))
        (tmp_path / "summary.md").write_text("This is the summary.")

        step = StoreMarkdownStep()
        outputs = step.execute(tmp_path / "audio.wav", tmp_path)

        content = outputs[0].read_text()
        assert "## Summary" in content
        assert "This is the summary." in content


# =============================================================================
# Test store_sqlite
# =============================================================================


class TestStoreSqlite:
    """Test store_sqlite step."""

    def test_init_db_creates_tables(self):
        """_init_db creates expected tables."""
        from infomux.steps.store_sqlite import _init_db
        import sqlite3

        conn = sqlite3.connect(":memory:")
        _init_db(conn)

        # Check tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}

        assert "runs" in tables
        assert "segments" in tables
        assert "summaries" in tables
        # FTS5 virtual table
        assert "transcripts" in tables

        conn.close()

    def test_store_run_inserts_data(self, tmp_path):
        """_store_run inserts run data correctly."""
        from infomux.steps.store_sqlite import _init_db, _store_run
        import sqlite3

        # Setup
        job = {
            "id": "run-sqlite-test",
            "created_at": "2026-01-10",
            "status": "completed",
            "steps": [],
        }
        (tmp_path / "job.json").write_text(json.dumps(job))
        (tmp_path / "transcript.txt").write_text("Test transcript")

        conn = sqlite3.connect(":memory:")
        _init_db(conn)

        # Execute
        run_id = _store_run(conn, tmp_path)
        conn.commit()

        assert run_id == "run-sqlite-test"

        # Verify data
        cursor = conn.execute("SELECT id, status FROM runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        assert row[0] == "run-sqlite-test"
        assert row[1] == "completed"

        # Verify transcript
        cursor = conn.execute(
            "SELECT content FROM transcripts WHERE run_id = ?", (run_id,)
        )
        row = cursor.fetchone()
        assert row[0] == "Test transcript"

        conn.close()

    @patch("infomux.steps.store_sqlite._get_db_path")
    def test_execute_uses_db_path(self, mock_get_db_path, tmp_path):
        """Execute uses configured database path."""
        from infomux.steps.store_sqlite import StoreSqliteStep

        db_path = tmp_path / "test.db"
        mock_get_db_path.return_value = db_path

        job = {"id": "run-123", "created_at": "2026-01-10", "steps": []}
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "job.json").write_text(json.dumps(job))

        step = StoreSqliteStep()
        step.execute(run_dir / "audio.wav", run_dir)

        assert db_path.exists()


# =============================================================================
# Test store_s3
# =============================================================================


class TestStoreS3:
    """Test store_s3 step."""

    def test_get_s3_config_requires_bucket(self, monkeypatch):
        """Raises StepError if INFOMUX_S3_BUCKET not set."""
        from infomux.steps.store_s3 import _get_s3_config
        from infomux.steps import StepError

        monkeypatch.delenv("INFOMUX_S3_BUCKET", raising=False)

        with pytest.raises(StepError) as exc:
            _get_s3_config()
        assert "INFOMUX_S3_BUCKET" in str(exc.value)

    def test_get_s3_config_returns_values(self, monkeypatch):
        """Returns bucket and prefix from environment."""
        from infomux.steps.store_s3 import _get_s3_config

        monkeypatch.setenv("INFOMUX_S3_BUCKET", "my-bucket")
        monkeypatch.setenv("INFOMUX_S3_PREFIX", "transcripts/")

        bucket, prefix = _get_s3_config()
        assert bucket == "my-bucket"
        assert prefix == "transcripts/"

    def test_get_content_type(self):
        """Returns correct content types for files."""
        from infomux.steps.store_s3 import _get_content_type

        assert _get_content_type(Path("file.json")) == "application/json"
        assert _get_content_type(Path("file.txt")) == "text/plain"
        assert _get_content_type(Path("file.wav")) == "audio/wav"
        assert _get_content_type(Path("file.xyz")) == "application/octet-stream"

    def test_upload_to_s3_calls_client(self, tmp_path):
        """_upload_to_s3 calls s3.upload_file for each file."""
        # Import boto3 here to check if it's available
        pytest.importorskip("boto3")

        from infomux.steps.store_s3 import _upload_to_s3

        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.json").write_text("{}")

        # Mock the boto3 client after import
        mock_client = MagicMock()
        with patch("boto3.client", return_value=mock_client):
            uploaded = _upload_to_s3(tmp_path, "my-bucket", "prefix/")

        assert len(uploaded) == 2
        assert mock_client.upload_file.call_count == 2

    @patch("infomux.steps.store_s3._get_s3_config")
    @patch("infomux.steps.store_s3._upload_to_s3")
    def test_execute_uploads_run_dir(self, mock_upload, mock_config, tmp_path):
        """Execute uploads run directory contents."""
        from infomux.steps.store_s3 import StoreS3Step

        mock_config.return_value = ("bucket", "prefix/")
        mock_upload.return_value = ["key1", "key2"]

        step = StoreS3Step()
        outputs = step.execute(tmp_path / "audio.wav", tmp_path)

        assert outputs == []  # No local outputs
        mock_upload.assert_called_once_with(tmp_path, "bucket", "prefix/")


# =============================================================================
# Test store_postgres
# =============================================================================


class TestStorePostgres:
    """Test store_postgres step."""

    def test_get_postgres_url_requires_env(self, monkeypatch):
        """Raises StepError if INFOMUX_POSTGRES_URL not set."""
        from infomux.steps.store_postgres import _get_postgres_url
        from infomux.steps import StepError

        monkeypatch.delenv("INFOMUX_POSTGRES_URL", raising=False)

        with pytest.raises(StepError) as exc:
            _get_postgres_url()
        assert "INFOMUX_POSTGRES_URL" in str(exc.value)

    def test_get_postgres_url_returns_value(self, monkeypatch):
        """Returns URL from environment."""
        from infomux.steps.store_postgres import _get_postgres_url

        monkeypatch.setenv("INFOMUX_POSTGRES_URL", "postgresql://user:pass@host/db")

        url = _get_postgres_url()
        assert url == "postgresql://user:pass@host/db"

    def test_execute_connects_and_stores(self, tmp_path, monkeypatch):
        """Execute connects to postgres and stores data."""
        # Skip if psycopg2 not installed
        psycopg2 = pytest.importorskip("psycopg2")

        from infomux.steps.store_postgres import StorePostgresStep

        monkeypatch.setenv("INFOMUX_POSTGRES_URL", "postgresql://test")

        # Setup mock connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Setup run dir
        job = {"id": "run-pg-test", "created_at": "2026-01-10", "steps": []}
        (tmp_path / "job.json").write_text(json.dumps(job))

        step = StorePostgresStep()

        with patch("psycopg2.connect", return_value=mock_conn):
            outputs = step.execute(tmp_path / "audio.wav", tmp_path)

        assert outputs == []
        mock_conn.commit.assert_called()
        mock_conn.close.assert_called()


# =============================================================================
# Test store_obsidian
# =============================================================================


class TestStoreObsidian:
    """Test store_obsidian step."""

    def test_get_obsidian_config_requires_vault(self, monkeypatch):
        """Raises StepError if INFOMUX_OBSIDIAN_VAULT not set."""
        from infomux.steps.store_obsidian import _get_obsidian_config
        from infomux.steps import StepError

        monkeypatch.delenv("INFOMUX_OBSIDIAN_VAULT", raising=False)

        with pytest.raises(StepError) as exc:
            _get_obsidian_config()
        assert "INFOMUX_OBSIDIAN_VAULT" in str(exc.value)

    def test_get_obsidian_config_returns_values(self, tmp_path, monkeypatch):
        """Returns vault path, folder, and tags."""
        from infomux.steps.store_obsidian import _get_obsidian_config

        monkeypatch.setenv("INFOMUX_OBSIDIAN_VAULT", str(tmp_path))
        monkeypatch.setenv("INFOMUX_OBSIDIAN_FOLDER", "Notes")
        monkeypatch.setenv("INFOMUX_OBSIDIAN_TAGS", "audio,meeting")

        vault, folder, tags = _get_obsidian_config()
        assert vault == tmp_path
        assert folder == "Notes"
        assert tags == ["audio", "meeting"]

    def test_sanitize_filename(self):
        """Sanitizes invalid characters from filenames."""
        from infomux.steps.store_obsidian import _sanitize_filename

        assert _sanitize_filename("file:name?test") == "filenametest"
        assert _sanitize_filename("normal-file_name") == "normal-file_name"

    def test_generate_obsidian_note_has_frontmatter(self, tmp_path):
        """Generated note includes YAML frontmatter."""
        from infomux.steps.store_obsidian import _generate_obsidian_note

        data = RunData(
            run_id="run-123",
            created_at="2026-01-10T12:00:00",
            input_path="/test/file.mp4",
            pipeline="transcribe",
            transcript_text="Hello world",
        )

        note = _generate_obsidian_note(data, ["infomux", "test"])

        assert note.startswith("---")
        assert "date: 2026-01-10" in note
        assert "tags: [infomux, test]" in note
        assert "---" in note[3:]  # Closing frontmatter

    def test_execute_creates_note_in_vault(self, tmp_path, monkeypatch):
        """Execute creates note file in vault folder."""
        from infomux.steps.store_obsidian import StoreObsidianStep

        vault = tmp_path / "vault"
        vault.mkdir()
        monkeypatch.setenv("INFOMUX_OBSIDIAN_VAULT", str(vault))

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        job = {
            "id": "run-obs-test",
            "created_at": "2026-01-10T12:00:00",
            "steps": [],
            "input": {"path": "/test/my-file.mp4"},
        }
        (run_dir / "job.json").write_text(json.dumps(job))

        step = StoreObsidianStep()
        outputs = step.execute(run_dir / "audio.wav", run_dir)

        assert len(outputs) == 1
        assert outputs[0].parent.name == "Transcripts"
        assert "my-file" in outputs[0].name
        assert outputs[0].exists()


# =============================================================================
# Test store_bear
# =============================================================================


class TestStoreBear:
    """Test store_bear step."""

    def test_get_bear_tags_default(self, monkeypatch):
        """Returns default tags when not configured."""
        from infomux.steps.store_bear import _get_bear_tags

        monkeypatch.delenv("INFOMUX_BEAR_TAGS", raising=False)

        tags = _get_bear_tags()
        assert "infomux" in tags
        assert "transcript" in tags

    def test_get_bear_tags_from_env(self, monkeypatch):
        """Returns tags from environment."""
        from infomux.steps.store_bear import _get_bear_tags

        monkeypatch.setenv("INFOMUX_BEAR_TAGS", "meeting,notes,voice")

        tags = _get_bear_tags()
        assert tags == ["meeting", "notes", "voice"]

    def test_generate_bear_note_returns_title_and_body(self):
        """Generates title and body for Bear note."""
        from infomux.steps.store_bear import _generate_bear_note

        data = RunData(
            run_id="run-123",
            created_at="2026-01-10T12:00:00",
            input_path="/test/my-recording.mp4",
            transcript_text="This is the transcript.",
        )

        title, body = _generate_bear_note(data)

        assert title == "my-recording"
        assert "This is the transcript." in body
        assert "**Date:**" in body

    @patch("infomux.steps.store_bear.subprocess.run")
    def test_open_bear_url_calls_subprocess(self, mock_run):
        """_open_bear_url calls macOS open command."""
        from infomux.steps.store_bear import _open_bear_url

        mock_run.return_value = MagicMock(returncode=0)

        _open_bear_url("Test Title", "Test body", ["tag1", "tag2"])

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "open"
        assert "bear://x-callback-url/create" in call_args[1]

    @patch("infomux.steps.store_bear._open_bear_url")
    @patch("sys.platform", "darwin")
    def test_execute_creates_bear_note(self, mock_open_bear, tmp_path, monkeypatch):
        """Execute creates note via Bear URL scheme."""
        from infomux.steps.store_bear import StoreBearStep

        job = {"id": "run-bear-test", "created_at": "2026-01-10", "steps": []}
        (tmp_path / "job.json").write_text(json.dumps(job))

        step = StoreBearStep()

        # Mock platform check
        with patch("sys.platform", "darwin"):
            outputs = step.execute(tmp_path / "audio.wav", tmp_path)

        assert outputs == []
        mock_open_bear.assert_called_once()

    def test_execute_fails_on_non_macos(self, tmp_path):
        """Execute fails gracefully on non-macOS."""
        from infomux.steps.store_bear import StoreBearStep
        from infomux.steps import StepError

        job = {"id": "run-123", "created_at": "2026-01-10", "steps": []}
        (tmp_path / "job.json").write_text(json.dumps(job))

        step = StoreBearStep()

        with patch("sys.platform", "linux"):
            with pytest.raises(StepError) as exc:
                step.execute(tmp_path / "audio.wav", tmp_path)
            assert "macOS" in str(exc.value)
