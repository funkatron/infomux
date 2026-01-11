"""
The 'resume' command: continue an interrupted or failed run.

This command loads an existing job envelope and resumes execution
from where it left off. Only steps that haven't completed will be run.

Usage:
    infomux resume <run-id>
    infomux resume --from-step transcribe <run-id>
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace

from infomux.job import JobStatus
from infomux.log import get_logger
from infomux.storage import load_job, run_exists, save_job

logger = get_logger(__name__)


def configure_parser(parser: ArgumentParser) -> None:
    """
    Configure the argument parser for the 'resume' command.

    Args:
        parser: The subparser to configure.
    """
    parser.add_argument(
        "run_id",
        type=str,
        help="ID of the run to resume",
    )
    parser.add_argument(
        "--from-step",
        type=str,
        default=None,
        help="Resume from a specific step (re-runs that step and all following)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )


def execute(args: Namespace) -> int:
    """
    Execute the 'resume' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
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

    logger.info("resuming run: %s", job.id)
    logger.debug("current status: %s", job.status)

    # Check if the run can be resumed
    if job.status == JobStatus.COMPLETED.value:
        logger.error("run already completed: %s", run_id)
        return 1

    if job.status == JobStatus.RUNNING.value:
        logger.warning("run appears to be in progress - proceeding anyway")

    # Find steps to resume
    steps_to_run = _get_resumable_steps(job, args.from_step)

    if not steps_to_run:
        logger.info("no steps to resume")
        return 0

    logger.info("steps to resume: %s", [s.name for s in steps_to_run])

    if args.dry_run:
        logger.info("dry run mode - not executing")
        for step in steps_to_run:
            print(f"would run: {step.name}", file=sys.stdout)
        return 0

    # Update status and save
    job.update_status(JobStatus.RUNNING)
    save_job(job)

    # TODO: Execute remaining pipeline steps
    # This is a stub - actual step execution will be implemented later
    logger.info("pipeline resumption not yet implemented")

    # Mark as completed (stub behavior)
    job.update_status(JobStatus.COMPLETED)
    save_job(job)

    # Output run ID to stdout for scripting
    print(job.id, file=sys.stdout)

    logger.info("run resumed and completed: %s", job.id)
    return 0


def _get_resumable_steps(job, from_step: str | None) -> list:
    """
    Determine which steps need to be run.

    Args:
        job: The JobEnvelope to analyze.
        from_step: Optional step name to resume from.

    Returns:
        List of StepRecords that should be executed.
    """
    if not job.steps:
        # No steps recorded - would need to run full pipeline
        logger.debug("no steps recorded in job")
        return []

    steps_to_run = []
    should_include = from_step is None  # If no from_step, start from first incomplete

    for step in job.steps:
        # If from_step is specified, start including from that step
        if from_step and step.name == from_step:
            should_include = True

        # If from_step is not specified, include incomplete steps
        if from_step is None and step.status not in ("completed",):
            should_include = True

        if should_include:
            steps_to_run.append(step)

    return steps_to_run
