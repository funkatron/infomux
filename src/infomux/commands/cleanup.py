"""
The 'cleanup' command: remove orphaned or unwanted runs.

This command helps clean up the runs directory by identifying and
removing orphaned runs (missing job.json), stuck runs (status: running),
or runs matching specific criteria.

Usage:
    infomux cleanup --dry-run              # Preview what would be deleted
    infomux cleanup --orphaned             # Delete runs without job.json
    infomux cleanup --status running       # Delete runs with specific status
    infomux cleanup --older-than 30d       # Delete runs older than 30 days
    infomux cleanup --force                # Actually delete (required)
"""

from __future__ import annotations

import shutil
import sys
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime, timedelta

from infomux.log import get_logger
from infomux.storage import get_runs_dir, load_job

logger = get_logger(__name__)


def configure_parser(parser: ArgumentParser) -> None:
    """
    Configure the argument parser for the 'cleanup' command.

    Args:
        parser: The subparser to configure.
    """
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting. "
        "Always use this first to see what would be removed. Shows run IDs and reasons.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually delete runs. Required to perform deletion (unless using --dry-run). "
        "This is a safety measure to prevent accidental deletion.",
    )
    parser.add_argument(
        "--orphaned",
        action="store_true",
        help="Delete runs without valid job.json files. "
        "These are typically incomplete or corrupted runs that can't be resumed or inspected.",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["pending", "running", "failed", "interrupted", "completed"],
        help="Delete runs with specific status. "
        "Common use: --status running (to clean up stuck/interrupted runs). "
        "Use with --older-than for safety (e.g., delete failed runs older than 7 days).",
    )
    parser.add_argument(
        "--older-than",
        type=str,
        metavar="N[d|w|m]",
        help="Delete runs older than specified time. "
        "Format: number followed by unit (d=days, w=weeks, m=months). "
        "Examples: 30d (30 days), 2w (2 weeks), 1m (1 month). "
        "Can be combined with --status for more targeted cleanup.",
    )
    parser.add_argument(
        "--min-age",
        type=str,
        metavar="N[d|w|m]",
        help="Minimum age for runs to be considered (safety check). "
        "Prevents deletion of very recent runs even if they match other criteria. "
        "Example: --min-age 1d (only delete runs at least 1 day old). "
        "Useful when combining with --status or --older-than.",
    )


def parse_time_spec(spec: str) -> timedelta:
    """
    Parse a time specification like "30d", "2w", "1m".

    Args:
        spec: Time specification string.

    Returns:
        Timedelta object.

    Raises:
        ValueError: If the specification is invalid.
    """
    if not spec:
        raise ValueError("time specification cannot be empty")

    # Extract number and unit
    spec = spec.strip().lower()
    if spec[-1] not in "dwm":
        raise ValueError(f"invalid time unit: {spec[-1]} (expected d, w, or m)")

    try:
        number = int(spec[:-1])
    except ValueError:
        raise ValueError(f"invalid number in time specification: {spec[:-1]}")

    unit = spec[-1]
    if unit == "d":
        return timedelta(days=number)
    elif unit == "w":
        return timedelta(weeks=number)
    elif unit == "m":
        return timedelta(days=number * 30)  # Approximate month as 30 days
    else:
        raise ValueError(f"unknown time unit: {unit}")


def execute(args: Namespace) -> int:
    """
    Execute the 'cleanup' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Require either --dry-run or --force
    if not args.dry_run and not args.force:
        logger.error("either --dry-run or --force is required")
        logger.error("use --dry-run to preview what would be deleted")
        return 1

    # Require at least one filter
    if not any([args.orphaned, args.status, args.older_than]):
        logger.error("at least one filter is required: --orphaned, --status, or --older-than")
        return 1

    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        logger.info("runs directory does not exist: %s", runs_dir)
        return 0

    # Parse time specifications
    older_than_delta = None
    if args.older_than:
        try:
            older_than_delta = parse_time_spec(args.older_than)
        except ValueError as e:
            logger.error("invalid --older-than specification: %s", e)
            return 1

    min_age_delta = None
    if args.min_age:
        try:
            min_age_delta = parse_time_spec(args.min_age)
        except ValueError as e:
            logger.error("invalid --min-age specification: %s", e)
            return 1

    # Find runs to delete
    runs_to_delete: list[tuple[str, str]] = []  # (run_id, reason)

    # Check all directories in runs_dir
    for entry in runs_dir.iterdir():
        if not entry.is_dir():
            continue

        run_id = entry.name
        job_path = entry / "job.json"

        # Check for orphaned runs (no job.json)
        if args.orphaned:
            if not job_path.exists():
                runs_to_delete.append((run_id, "orphaned (no job.json)"))
                continue

        # If job.json exists, try to load it
        if job_path.exists():
            try:
                job = load_job(run_id)

                # Check status filter
                if args.status and job.status == args.status:
                    # Additional check: if min_age is set, verify the run is old enough
                    if min_age_delta:
                        try:
                            created_str = job.created_at.replace("Z", "+00:00")
                            created_at = datetime.fromisoformat(created_str)
                            if created_at.tzinfo is None:
                                created_at = created_at.replace(tzinfo=UTC)
                            now = datetime.now(UTC)
                            age = now - created_at
                            if age < min_age_delta:
                                logger.debug("skipping %s: too recent (age: %s)", run_id, age)
                                continue
                        except (ValueError, AttributeError) as e:
                            logger.debug("could not parse date for %s: %s", run_id, e)
                            # If we can't parse the date, skip the min_age check
                            pass

                    runs_to_delete.append((run_id, f"status: {job.status}"))
                    continue

                # Check age filter
                if older_than_delta:
                    try:
                        # Parse ISO format timestamp
                        created_str = job.created_at.replace("Z", "+00:00")
                        created_at = datetime.fromisoformat(created_str)
                        # Ensure timezone-aware
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=UTC)
                        # Get current time in UTC
                        now = datetime.now(UTC)
                        age = now - created_at
                        if age > older_than_delta:
                            # Additional check: if min_age is set, verify the run is old enough
                            if min_age_delta and age < min_age_delta:
                                logger.debug("skipping %s: too recent (age: %s)", run_id, age)
                                continue

                            runs_to_delete.append((run_id, f"older than {args.older_than} (age: {age.days}d)"))
                    except (ValueError, AttributeError) as e:
                        logger.debug("could not parse date for %s: %s", run_id, e)
                        continue

            except Exception as e:
                # If we can't load the job, it might be corrupted
                if args.orphaned:
                    runs_to_delete.append((run_id, f"orphaned (corrupted job.json: {e})"))
                else:
                    logger.debug("skipping %s: could not load job: %s", run_id, e)

    # Report findings
    if not runs_to_delete:
        logger.info("no runs found matching the specified criteria")
        return 0

    # Sort by run_id for consistent output
    runs_to_delete.sort(key=lambda x: x[0])

    if args.dry_run:
        print(f"Would delete {len(runs_to_delete)} run(s):", file=sys.stdout)
        print()
        for run_id, reason in runs_to_delete:
            print(f"  {run_id} ({reason})", file=sys.stdout)
        print()
        print("Run with --force to actually delete these runs.", file=sys.stdout)
        return 0

    # Actually delete
    deleted_count = 0
    failed_count = 0

    for run_id, reason in runs_to_delete:
        run_dir = runs_dir / run_id
        try:
            logger.info("deleting %s (%s)", run_id, reason)
            shutil.rmtree(run_dir)
            deleted_count += 1
        except Exception as e:
            logger.error("failed to delete %s: %s", run_id, e)
            failed_count += 1

    # Summary
    print(f"Deleted {deleted_count} run(s)", file=sys.stdout)
    if failed_count > 0:
        print(f"Failed to delete {failed_count} run(s)", file=sys.stderr)
        return 1

    return 0
