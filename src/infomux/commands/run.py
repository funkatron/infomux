"""
The 'run' command: process a media file through the pipeline.

This command takes a media file as input, creates a new job envelope,
and runs it through the configured pipeline steps. Each step produces
artifacts that are stored in the run directory.

Usage:
    infomux run input.mp4
    infomux run --pipeline transcribe input.mp4
    infomux run --steps extract_audio input.mp4
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from infomux.config import get_tool_paths
from infomux.job import InputFile, JobEnvelope, JobStatus
from infomux.log import get_logger
from infomux.pipeline import run_pipeline
from infomux.pipeline_def import get_pipeline, list_pipelines
from infomux.storage import get_run_dir, save_job

logger = get_logger(__name__)


def configure_parser(parser: ArgumentParser) -> None:
    """
    Configure the argument parser for the 'run' command.

    Args:
        parser: The subparser to configure.
    """
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",  # Optional when using --check-deps or --list-pipelines
        help="Path to the input media file",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        default=None,
        help="Pipeline to run (default: transcribe)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of steps to run (subset of pipeline)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check for required dependencies and exit",
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List available pipelines and exit",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Ollama model for summarization (e.g., qwen2.5:32b-instruct)",
    )


def execute(args: Namespace) -> int:
    """
    Execute the 'run' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # List pipelines mode
    if args.list_pipelines:
        return _list_pipelines()

    # Check dependencies mode
    if args.check_deps:
        return _check_dependencies()

    # Require input file for normal run
    if not args.input:
        logger.error("input file is required")
        return 1

    input_path: Path = args.input

    # Validate input file
    if not input_path.exists():
        logger.error("input file not found: %s", input_path)
        return 1

    if not input_path.is_file():
        logger.error("input path is not a file: %s", input_path)
        return 1

    logger.info("processing input: %s", input_path)

    # Get pipeline definition
    try:
        pipeline = get_pipeline(args.pipeline)
    except ValueError as e:
        logger.error(str(e))
        return 1

    if args.pipeline is None:
        logger.info(
            "using default pipeline '%s' (%s)",
            pipeline.name,
            " → ".join(pipeline.step_names()),
        )
    else:
        logger.info("using pipeline: %s", pipeline.name)

    # Create input file metadata
    try:
        input_file = InputFile.from_path(input_path)
        logger.debug("input sha256: %s", input_file.sha256)
        logger.debug("input size: %d bytes", input_file.size_bytes)
    except Exception as e:
        logger.error("failed to read input file: %s", e)
        return 1

    # Create job envelope
    job = JobEnvelope.create(input_file=input_file)

    # Parse steps if specified (subset of pipeline)
    step_names = None
    if args.steps:
        step_names = [s.strip() for s in args.steps.split(",")]
        # Validate steps exist in pipeline
        valid_steps = set(pipeline.step_names())
        invalid = [s for s in step_names if s not in valid_steps]
        if invalid:
            logger.error(
                "unknown steps: %s (available: %s)",
                invalid,
                list(valid_steps),
            )
            return 1
        logger.info("running steps: %s", step_names)

    if args.dry_run:
        logger.info("dry run mode - not executing")
        # Show pipeline info
        print(f"Pipeline: {pipeline.name}")
        print(f"Description: {pipeline.description}")
        print(f"Steps: {pipeline.step_names()}")
        if step_names:
            print(f"Selected: {step_names}")
        print()
        print(job.to_json())
        return 0

    # Validate dependencies before starting
    tools = get_tool_paths()
    errors = tools.validate()
    if errors:
        for error in errors:
            logger.error(error)
        logger.error("run 'infomux run --check-deps' for more information")
        return 1

    # Save the job envelope
    job.update_status(JobStatus.RUNNING)
    save_job(job)
    run_dir = get_run_dir(job.id)
    logger.info("created run: %s", job.id)
    logger.debug("run directory: %s", run_dir)

    # Set model override if specified
    if args.model:
        import os
        os.environ["INFOMUX_OLLAMA_MODEL"] = args.model
        logger.debug("using model: %s", args.model)

    # Execute pipeline
    success = run_pipeline(job, run_dir, pipeline=pipeline, step_names=step_names)

    # Update final status
    if success:
        job.update_status(JobStatus.COMPLETED)
        logger.info("run completed: %s", job.id)
    else:
        # Status already set by pipeline on failure
        logger.error("run failed: %s", job.id)

    # Save final state
    save_job(job)

    # Output run directory path to stdout for scripting
    print(run_dir, file=sys.stdout)

    return 0 if success else 1


def _list_pipelines() -> int:
    """
    List available pipelines.

    Returns:
        Exit code (always 0).
    """
    print("Available pipelines:")
    print()
    for name in list_pipelines():
        pipeline = get_pipeline(name)
        print(f"  {name}")
        print(f"    {pipeline.description}")
        print(f"    Steps: {' → '.join(pipeline.step_names())}")
        print()
    return 0


def _check_dependencies() -> int:
    """
    Check for required external dependencies.

    Returns:
        Exit code (0 if all deps found, 1 if any missing).
    """
    tools = get_tool_paths()

    print("Checking dependencies...")
    print()

    # ffmpeg
    if tools.ffmpeg:
        print(f"✓ ffmpeg: {tools.ffmpeg}")
    else:
        print("✗ ffmpeg: NOT FOUND")
        print("  Install: brew install ffmpeg")

    # whisper-cli
    if tools.whisper_cli:
        print(f"✓ whisper-cli: {tools.whisper_cli}")
    else:
        print("✗ whisper-cli: NOT FOUND")
        print("  Install: brew install whisper-cpp")

    # whisper model
    if tools.whisper_model:
        size_mb = tools.whisper_model.stat().st_size / (1024 * 1024)
        print(f"✓ whisper model: {tools.whisper_model} ({size_mb:.1f} MB)")
    else:
        print("✗ whisper model: NOT FOUND")
        print("  Download:")
        print("    mkdir -p ~/.local/share/infomux/models/whisper")
        print("    curl -L -o ~/.local/share/infomux/models/whisper/ggml-base.en.bin")
        print("      https://huggingface.co/.../ggml-base.en.bin")

    print()

    errors = tools.validate()
    if errors:
        print(f"Missing {len(errors)} dependency(ies)")
        return 1
    else:
        print("All dependencies found!")
        return 0
