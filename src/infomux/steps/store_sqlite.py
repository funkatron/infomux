"""
Store step: persist transcripts and metadata to SQLite.

Creates a searchable database with full-text search (FTS5) for finding
content across all runs.

Database location: ~/.local/share/infomux/infomux.db

Tables:
    - runs: Run metadata (id, created_at, input_path, pipeline, status)
    - transcripts: Full transcript text with FTS5 search
    - segments: Individual transcript segments with timestamps
    - summaries: LLM-generated summaries
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step

logger = get_logger(__name__)

# Database location
DEFAULT_DB_PATH = Path.home() / ".local/share/infomux/infomux.db"

# Output filename (we don't create files, just update DB)
STORE_SQLITE_FILENAME = None  # No file output


def _get_db_path() -> Path:
    """Get database path, respecting INFOMUX_DATA_DIR if set."""
    import os

    data_dir = os.environ.get("INFOMUX_DATA_DIR")
    if data_dir:
        return Path(data_dir) / "infomux.db"
    return DEFAULT_DB_PATH


def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize database schema."""
    conn.executescript("""
        -- Run metadata
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            input_path TEXT,
            input_hash TEXT,
            pipeline TEXT,
            status TEXT,
            duration_seconds REAL,
            job_json TEXT
        );

        -- Full transcript text with FTS5 for searching
        CREATE VIRTUAL TABLE IF NOT EXISTS transcripts USING fts5(
            run_id,
            content,
            tokenize='porter unicode61'
        );

        -- Individual segments with timestamps
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            start_ms INTEGER,
            end_ms INTEGER,
            text TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );

        -- Summaries
        CREATE TABLE IF NOT EXISTS summaries (
            run_id TEXT PRIMARY KEY,
            content TEXT,
            model TEXT,
            created_at TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );

        -- Index for faster lookups
        CREATE INDEX IF NOT EXISTS idx_segments_run ON segments(run_id);
    """)


def _store_run(conn: sqlite3.Connection, run_dir: Path) -> str | None:
    """
    Store run data in the database.

    Args:
        conn: Database connection.
        run_dir: Path to the run directory.

    Returns:
        Run ID if successful, None otherwise.
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

    # Insert or update run
    conn.execute(
        """
        INSERT OR REPLACE INTO runs
        (id, created_at, input_path, input_hash, pipeline, status, job_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            job.get("created_at"),
            job.get("input", {}).get("path"),
            job.get("input", {}).get("sha256"),
            job.get("config", {}).get("pipeline"),
            job.get("status"),
            json.dumps(job),
        ),
    )

    # Store transcript if exists
    transcript_path = run_dir / "transcript.txt"
    if transcript_path.exists():
        content = transcript_path.read_text()

        # Delete old transcript for this run
        conn.execute("DELETE FROM transcripts WHERE run_id = ?", (run_id,))

        # Insert new transcript
        conn.execute(
            "INSERT INTO transcripts (run_id, content) VALUES (?, ?)",
            (run_id, content),
        )

    # Store segments from JSON if exists
    json_path = run_dir / "transcript.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)

            # Delete old segments
            conn.execute("DELETE FROM segments WHERE run_id = ?", (run_id,))

            # Parse whisper.cpp JSON format
            transcription = data.get("transcription", [])
            for segment in transcription:
                offsets = segment.get("offsets", {})
                conn.execute(
                    """
                    INSERT INTO segments (run_id, start_ms, end_ms, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        offsets.get("from"),
                        offsets.get("to"),
                        segment.get("text", "").strip(),
                    ),
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse transcript.json: %s", e)

    # Store summary if exists
    summary_path = run_dir / "summary.md"
    if summary_path.exists():
        content = summary_path.read_text()

        # Get model info from job
        model = None
        for step in job.get("steps", []):
            if step.get("name") == "summarize" and step.get("model_info"):
                model = step["model_info"].get("model", {}).get("name")
                break

        conn.execute(
            """
            INSERT OR REPLACE INTO summaries (run_id, content, model, created_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (run_id, content, model),
        )

    return run_id


@register_step
@dataclass
class StoreSqliteStep:
    """
    Pipeline step to store run data in SQLite.

    Creates a searchable database of all transcripts, segments, and summaries.
    Enables full-text search across all runs.
    """

    name: str = "store_sqlite"

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Store run data in SQLite database.

        Args:
            input_path: Not used (we read from output_dir).
            output_dir: The run directory containing job.json and artifacts.

        Returns:
            Empty list (no file outputs, data goes to database).

        Raises:
            StepError: If database operation fails.
        """
        db_path = _get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("storing run in database: %s", db_path)

        try:
            conn = sqlite3.connect(str(db_path))
            _init_db(conn)

            run_id = _store_run(conn, output_dir)
            if run_id:
                conn.commit()
                logger.info("stored run: %s", run_id)
            else:
                logger.warning("no run data to store")

            conn.close()
            return []

        except sqlite3.Error as e:
            raise StepError(self.name, f"database error: {e}")


def run(input_path: Path, output_dir: Path) -> StepResult:
    """
    Run the store_sqlite step.

    Args:
        input_path: Not used.
        output_dir: Run directory to store.

    Returns:
        StepResult with execution details.
    """
    step = StoreSqliteStep()
    start_time = time.monotonic()

    try:
        outputs = step.execute(input_path, output_dir)
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=True,
            outputs=outputs,
            duration_seconds=duration,
        )
    except StepError as e:
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=False,
            outputs=[],
            duration_seconds=duration,
            error=str(e),
        )


# Utility functions for querying the database


def search_transcripts(query: str, limit: int = 10) -> list[dict]:
    """
    Search transcripts using full-text search.

    Args:
        query: Search query (supports FTS5 syntax).
        limit: Maximum results to return.

    Returns:
        List of matching runs with snippets.
    """
    db_path = _get_db_path()
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    results = conn.execute(
        """
        SELECT
            t.run_id,
            r.input_path,
            r.created_at,
            snippet(transcripts, 1, '>>>', '<<<', '...', 32) as snippet
        FROM transcripts t
        JOIN runs r ON t.run_id = r.id
        WHERE transcripts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()

    conn.close()
    return [dict(r) for r in results]


def get_run_segments(run_id: str) -> list[dict]:
    """
    Get all segments for a run.

    Args:
        run_id: The run ID.

    Returns:
        List of segments with timestamps.
    """
    db_path = _get_db_path()
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    results = conn.execute(
        """
        SELECT start_ms, end_ms, text
        FROM segments
        WHERE run_id = ?
        ORDER BY start_ms
        """,
        (run_id,),
    ).fetchall()

    conn.close()
    return [dict(r) for r in results]
