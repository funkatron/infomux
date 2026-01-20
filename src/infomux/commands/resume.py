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

from infomux.config import get_tool_paths
from infomux.job import JobStatus
from infomux.log import get_logger
from infomux.pipeline import get_resumable_steps, run_pipeline
from infomux.pipeline_def import get_pipeline
from infomux.storage import get_run_dir, load_job, run_exists, save_job

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
        help="ID of the run to resume (e.g., run-20260111-020549-c36c19). "
        "Use 'infomux inspect --list' to see all available run IDs.",
    )
    parser.add_argument(
        "--from-step",
        type=str,
        default=None,
        help="Resume from a specific step, re-running that step and all following steps. "
        "Completed steps before this are skipped. Example: --from-step transcribe",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing. "
        "Displays which steps would be re-run and their configuration.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Ollama model for summarization steps. "
        "Overrides the model used in the original run. Example: qwen2.5:32b-instruct",
    )
    parser.add_argument(
        "--content-type-hint",
        type=str,
        default=None,
        metavar="TYPE",
        help="Hint for content type to improve summarization quality. "
        "Options: meeting, talk, podcast, lecture, standup, 1on1, or any custom string. "
        "Overrides the hint from the original run.",
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
        if not args.from_step:
            logger.error("run already completed: %s", run_id)
            logger.info("use --from-step to re-run specific steps")
            return 1
        else:
            logger.info("re-running steps from '%s' on completed run", args.from_step)

    if job.status == JobStatus.RUNNING.value:
        logger.warning("run appears to be in progress - proceeding anyway")

    # Get the pipeline used for this run
    pipeline_name = job.config.get("pipeline")
    try:
        pipeline = get_pipeline(pipeline_name)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Determine which steps to run
    steps_to_run = get_resumable_steps(job, args.from_step)

    if not steps_to_run:
        logger.info("no steps to resume")
        return 0

    logger.info("steps to resume: %s", steps_to_run)

    if args.dry_run:
        logger.info("dry run mode - not executing")
        print(f"Pipeline: {pipeline.name}")
        print(f"Steps to run: {' → '.join(steps_to_run)}")
        return 0

    # Validate dependencies before resuming
    tools = get_tool_paths()
    errors = tools.validate()
    if errors:
        for error in errors:
            logger.error(error)
        logger.error("run 'infomux run --check-deps' for more information")
        return 1

    # Clear failed/incomplete step records that will be re-run
    # Keep completed steps that won't be re-run
    job.steps = [s for s in job.steps if s.name not in steps_to_run]

    # Update status and save
    job.update_status(JobStatus.RUNNING)
    job.error = None  # Clear previous error
    save_job(job)

    run_dir = get_run_dir(job.id)
    logger.debug("run directory: %s", run_dir)

    # Set model override if specified
    if args.model:
        import os
        os.environ["INFOMUX_OLLAMA_MODEL"] = args.model
        logger.debug("using model: %s", args.model)

    # Set content type hint if specified
    if args.content_type_hint:
        import os
        os.environ["INFOMUX_CONTENT_TYPE_HINT"] = args.content_type_hint
        logger.debug("content type hint: %s", args.content_type_hint)

    # Execute the remaining steps
    success = run_pipeline(job, run_dir, pipeline=pipeline, step_names=steps_to_run)

    # Update final status
    if success:
        job.update_status(JobStatus.COMPLETED)
        logger.info("run resumed and completed: %s", job.id)
    else:
        # Status already set by pipeline on failure
        logger.error("run failed: %s", job.id)

    # Save final state
    save_job(job)

    # Output run directory path to stdout for scripting
    print(get_run_dir(job.id), file=sys.stdout)

    return 0 if success else 1
