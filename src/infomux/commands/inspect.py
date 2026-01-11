"""
The 'inspect' command: display details about a run.

This command shows the job envelope and status of a completed or
in-progress run. It's useful for debugging and auditing.

Usage:
    infomux inspect <run-id>
    infomux inspect --json <run-id>
    infomux inspect --list
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace

from infomux.log import get_logger
from infomux.storage import list_runs, load_job, run_exists

logger = get_logger(__name__)


def configure_parser(parser: ArgumentParser) -> None:
    """
    Configure the argument parser for the 'inspect' command.

    Args:
        parser: The subparser to configure.
    """
    parser.add_argument(
        "run_id",
        type=str,
        nargs="?",
        default=None,
        help="ID of the run to inspect",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (default: human-readable)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        dest="list_runs",
        help="List all runs instead of inspecting one",
    )


def execute(args: Namespace) -> int:
    """
    Execute the 'inspect' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # List mode
    if args.list_runs:
        runs = list_runs()
        if not runs:
            logger.info("no runs found")
            return 0

        for run_id in runs:
            print(run_id, file=sys.stdout)
        return 0

    # Inspect mode requires a run ID
    if not args.run_id:
        logger.error("run_id is required (or use --list)")
        return 1

    run_id = args.run_id

    # Check if run exists
    if not run_exists(run_id):
        logger.error("run not found: %s", run_id)
        return 1

    # Load the job envelope
    try:
        job = load_job(run_id)
    except Exception as e:
        logger.error("failed to load run: %s", e)
        return 1

    # Output
    if args.json:
        print(job.to_json(), file=sys.stdout)
    else:
        # Human-readable format
        _print_job_summary(job)

    return 0


def _print_job_summary(job) -> None:
    """
    Print a human-readable summary of a job.

    Args:
        job: The JobEnvelope to summarize.
    """
    print(f"Run: {job.id}")
    print(f"Status: {job.status}")
    print(f"Created: {job.created_at}")
    print(f"Updated: {job.updated_at}")

    if job.input:
        print("\nInput:")
        print(f"  Path: {job.input.path}")
        print(f"  SHA256: {job.input.sha256[:16]}...")
        print(f"  Size: {job.input.size_bytes:,} bytes")

    if job.steps:
        print("\nSteps:")
        for step in job.steps:
            status_icon = {
                "pending": "○",
                "running": "◐",
                "completed": "●",
                "failed": "✗",
            }.get(step.status, "?")
            print(f"  {status_icon} {step.name}: {step.status}")
            if step.duration_seconds:
                print(f"      Duration: {step.duration_seconds:.2f}s")
            if step.error:
                print(f"      Error: {step.error}")

    if job.artifacts:
        print("\nArtifacts:")
        for artifact in job.artifacts:
            print(f"  - {artifact}")

    if job.error:
        print(f"\nError: {job.error}")
