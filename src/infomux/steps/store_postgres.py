"""
Store step: persist run data to PostgreSQL.

Stores transcripts, segments, and summaries in PostgreSQL for
multi-user access, advanced querying, and integration with other systems.

Environment variables:
    INFOMUX_POSTGRES_URL: Connection URL (required)
        Format: postgresql://user:pass@host:port/database

Requires: psycopg2 or psycopg

Output: No local files, data stored in PostgreSQL
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step
from infomux.steps.storage import RunData

logger = get_logger(__name__)

# No local output file
STORE_POSTGRES_FILENAME = None


def _get_postgres_url() -> str:
    """
    Get PostgreSQL connection URL from environment.

    Returns:
        Connection URL.

    Raises:
        StepError: If not configured.
    """
    url = os.environ.get("INFOMUX_POSTGRES_URL")
    if not url:
        raise StepError(
            "store_postgres",
            "INFOMUX_POSTGRES_URL not set. "
            "Set to: postgresql://user:pass@host:port/database",
        )
    return url


def _init_schema(conn) -> None:
    """Initialize PostgreSQL schema."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMPTZ,
                input_path TEXT,
                input_hash TEXT,
                pipeline TEXT,
                status TEXT,
                duration_seconds REAL,
                job_json JSONB
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                content TEXT,
                content_tsvector TSVECTOR
            );

            CREATE INDEX IF NOT EXISTS idx_transcripts_fts
                ON transcripts USING GIN(content_tsvector);

            CREATE TABLE IF NOT EXISTS segments (
                id SERIAL PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                start_ms INTEGER,
                end_ms INTEGER,
                text TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_segments_run ON segments(run_id);

            CREATE TABLE IF NOT EXISTS summaries (
                run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                content TEXT,
                model TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
    conn.commit()


def _store_run(conn, data: RunData) -> None:
    """Store run data in PostgreSQL."""
    import json

    with conn.cursor() as cur:
        # Upsert run
        cur.execute(
            """
            INSERT INTO runs (id, created_at, input_path, input_hash,
                            pipeline, status, duration_seconds, job_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                duration_seconds = EXCLUDED.duration_seconds,
                job_json = EXCLUDED.job_json
            """,
            (
                data.run_id,
                data.created_at,
                data.input_path,
                data.input_hash,
                data.pipeline,
                data.status,
                data.duration_seconds,
                json.dumps(data.job_json),
            ),
        )

        # Store transcript with tsvector for full-text search
        if data.transcript_text:
            cur.execute(
                """
                INSERT INTO transcripts (run_id, content, content_tsvector)
                VALUES (%s, %s, to_tsvector('english', %s))
                ON CONFLICT (run_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    content_tsvector = EXCLUDED.content_tsvector
                """,
                (data.run_id, data.transcript_text, data.transcript_text),
            )

        # Store segments
        if data.transcript_segments:
            cur.execute("DELETE FROM segments WHERE run_id = %s", (data.run_id,))
            for seg in data.transcript_segments:
                cur.execute(
                    """
                    INSERT INTO segments (run_id, start_ms, end_ms, text)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (data.run_id, seg.start_ms, seg.end_ms, seg.text),
                )

        # Store summary
        if data.summary:
            cur.execute(
                """
                INSERT INTO summaries (run_id, content, model)
                VALUES (%s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    model = EXCLUDED.model
                """,
                (data.run_id, data.summary, data.summary_model),
            )

    conn.commit()


@register_step
@dataclass
class StorePostgresStep:
    """
    Pipeline step to store run data in PostgreSQL.

    Creates searchable tables for:
    - runs: Metadata with JSONB for full job data
    - transcripts: Full-text search with tsvector
    - segments: Individual segments with timestamps
    - summaries: LLM summaries with model info

    Requires:
    - psycopg2 or psycopg installed
    - INFOMUX_POSTGRES_URL environment variable
    """

    name: str = "store_postgres"

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Store run data in PostgreSQL.

        Args:
            input_path: Not used.
            output_dir: The run directory.

        Returns:
            Empty list (no local outputs).

        Raises:
            StepError: If database operation fails.
        """
        try:
            import psycopg2
        except ImportError:
            try:
                import psycopg as psycopg2  # psycopg3
            except ImportError:
                raise StepError(
                    self.name,
                    "psycopg2 not installed. Run: pip install psycopg2-binary",
                )

        url = _get_postgres_url()
        run_data = RunData.from_run_dir(output_dir)
        if not run_data:
            raise StepError(self.name, "No run data found")

        logger.info("storing in PostgreSQL: %s", run_data.run_id)

        try:
            conn = psycopg2.connect(url)
            _init_schema(conn)
            _store_run(conn, run_data)
            conn.close()

            logger.info("stored run: %s", run_data.run_id)
            return []

        except psycopg2.Error as e:
            raise StepError(self.name, f"database error: {e}")
        except OSError as e:
            raise StepError(self.name, f"connection error: {e}")


def run(input_path: Path, output_dir: Path) -> StepResult:
    """Run the store_postgres step."""
    step = StorePostgresStep()
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


# Utility functions for querying


def search_transcripts(query: str, limit: int = 10) -> list[dict]:
    """
    Search transcripts using PostgreSQL full-text search.

    Args:
        query: Search query.
        limit: Maximum results.

    Returns:
        List of matching runs with snippets.
    """
    try:
        import psycopg2
    except ImportError:
        try:
            import psycopg as psycopg2
        except ImportError:
            return []

    url = os.environ.get("INFOMUX_POSTGRES_URL")
    if not url:
        return []

    try:
        conn = psycopg2.connect(url)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    t.run_id,
                    r.input_path,
                    r.created_at,
                    ts_headline('english', t.content, plainto_tsquery(%s),
                               'StartSel=>>>, StopSel=<<<') as snippet
                FROM transcripts t
                JOIN runs r ON t.run_id = r.id
                WHERE t.content_tsvector @@ plainto_tsquery(%s)
                ORDER BY ts_rank(t.content_tsvector, plainto_tsquery(%s)) DESC
                LIMIT %s
                """,
                (query, query, query, limit),
            )
            results = [
                {
                    "run_id": row[0],
                    "input_path": row[1],
                    "created_at": row[2],
                    "snippet": row[3],
                }
                for row in cur.fetchall()
            ]
        conn.close()
        return results
    except psycopg2.Error as e:
        logger.warning("PostgreSQL search failed: %s", e)
        return []
    except OSError as e:
        logger.warning("PostgreSQL connection failed: %s", e)
        return []
