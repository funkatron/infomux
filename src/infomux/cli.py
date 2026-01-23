"""
Command-line interface for infomux.

This module provides the main CLI entry point and subcommand routing.
stdout is reserved for machine-readable output; logs go to stderr.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from infomux import __version__
from infomux.commands import analyze_timing as analyze_timing_cmd
from infomux.commands import cleanup as cleanup_cmd
from infomux.commands import inspect as inspect_cmd
from infomux.commands import resume as resume_cmd
from infomux.commands import run as run_cmd
from infomux.commands import stream as stream_cmd
from infomux.log import configure_logging, get_logger

if TYPE_CHECKING:
    from argparse import Namespace

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for infomux.

    Returns:
        Configured ArgumentParser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="infomux",
        description="A local-first, deterministic media pipeline CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a media file
  infomux run input.mp4
  infomux run --pipeline summarize input.mp4
  infomux run https://example.com/audio.mp3

  # Real-time recording and transcription
  infomux stream
  infomux stream --device 0 --silence 5
  infomux stream --pipeline summarize

  # Inspect and manage runs
  infomux inspect --list
  infomux inspect run-20260111-020549-c36c19
  infomux inspect --open run-20260111-020549-c36c19
  infomux inspect --list-pipelines

  # Resume interrupted runs
  infomux resume run-20260111-020549-c36c19
  infomux resume --from-step transcribe run-20260111-020549-c36c19

  # Clean up old or orphaned runs
  infomux cleanup --dry-run --orphaned
  infomux cleanup --force --status running
  infomux cleanup --force --older-than 30d

Environment Variables:
  INFOMUX_DATA_DIR    Base directory for runs and models
                      (default: ~/.local/share/infomux)
  INFOMUX_LOG_LEVEL   Log verbosity: DEBUG, INFO, WARN, ERROR
                      (default: INFO)

For more information, see: https://github.com/funkatron/infomux
""",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase log verbosity (can be repeated)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        metavar="<command>",
    )

    # run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run a pipeline on a media file",
        description="Process a media file (or URL) through the configured pipeline steps. "
        "Supports audio, video, and text files. Automatically detects HTML content "
        "and uses the web-summarize pipeline when appropriate.",
        epilog="""
Examples:
  # Basic transcription (default pipeline)
  infomux run audio.mp3

  # Summarize with LLM
  infomux run --pipeline summarize meeting.mp4

  # Generate subtitles with timestamps
  infomux run --pipeline timed video.mp4

  # Process from URL
  infomux run https://example.com/podcast.mp3

  # Custom model for summarization
  infomux run --pipeline summarize --model qwen2.5:32b-instruct audio.mp3

  # Check dependencies
  infomux run --check-deps

For available pipelines, run: infomux inspect --list-pipelines
""",
    )
    run_cmd.configure_parser(run_parser)

    # inspect subcommand
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a previous run",
        description="Display details about a completed or in-progress run. "
        "Can also list all runs, available pipelines, and available steps. "
        "Useful for debugging, auditing, and discovering what's available.",
        epilog="""
Examples:
  # List all runs in tabular format
  infomux inspect --list

  # View details of a specific run
  infomux inspect run-20260111-020549-c36c19

  # Get JSON output for scripting
  infomux inspect --json run-20260111-020549-c36c19

  # Show path to run directory
  infomux inspect --path run-20260111-020549-c36c19

  # Open run directory in Finder (macOS) or file manager
  infomux inspect --open run-20260111-020549-c36c19

  # List available pipelines
  infomux inspect --list-pipelines

  # List available steps
  infomux inspect --list-steps
""",
    )
    inspect_cmd.configure_parser(inspect_parser)

    # resume subcommand
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an interrupted run",
        description="Continue a run that was interrupted or failed. "
        "Skips already-completed steps and re-runs from the specified point. "
        "Useful for recovering from errors or re-running steps with different settings.",
        epilog="""
Examples:
  # Resume a failed run (continues from where it stopped)
  infomux resume run-20260111-020549-c36c19

  # Re-run transcription step and all following steps
  infomux resume --from-step transcribe run-20260111-020549-c36c19

  # Re-generate summary with different model
  infomux resume --from-step summarize --model qwen2.5:32b-instruct run-20260111-020549-c36c19

  # Re-summarize with content type hint
  infomux resume --from-step summarize --content-type-hint meeting run-20260111-020549-c36c19

  # Preview what would be re-run
  infomux resume --dry-run run-20260111-020549-c36c19
""",
    )
    resume_cmd.configure_parser(resume_parser)

    # stream subcommand
    stream_parser = subparsers.add_parser(
        "stream",
        help="Real-time audio capture and transcription",
        description="Record from microphone and transcribe in real-time. "
        "Supports multiple stop conditions: duration, silence detection, or stop phrase. "
        "Can run additional pipelines (like summarize) after recording completes.",
        epilog="""
Examples:
  # Interactive device selection and recording
  infomux stream

  # Record from specific device
  infomux stream --device 2

  # 5-minute voice memo
  infomux stream --duration 300

  # Auto-stop after 5 seconds of silence (great for dictation)
  infomux stream --silence 5

  # Custom stop phrase
  infomux stream --stop-word "end note"

  # Record and summarize
  infomux stream --pipeline summarize

  # Meeting notes with auto-silence detection
  infomux stream --device 2 --silence 10 --pipeline summarize

  # List available audio devices
  infomux stream --list-devices
""",
    )
    stream_cmd.configure_parser(stream_parser)

    # cleanup subcommand
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Remove orphaned or unwanted runs",
        description="Clean up the runs directory by removing orphaned runs, stuck runs, or runs matching specific criteria. "
        "Always use --dry-run first to preview what would be deleted. Requires --force to actually delete.",
        epilog="""
Examples:
  # Preview orphaned runs (always do this first!)
  infomux cleanup --dry-run --orphaned

  # Delete orphaned runs (no valid job.json)
  infomux cleanup --force --orphaned

  # Delete stuck runs (status: running)
  infomux cleanup --force --status running

  # Delete runs older than 30 days
  infomux cleanup --force --older-than 30d

  # Delete failed runs older than 7 days (with safety check)
  infomux cleanup --force --status failed --older-than 7d --min-age 1d

  # Combine filters: delete orphaned and stuck runs
  infomux cleanup --force --orphaned --status running

Time specifications:
  Use 'd' for days, 'w' for weeks, 'm' for months
  Examples: 30d, 2w, 1m
""",
    )
    cleanup_cmd.configure_parser(cleanup_parser)

    # analyze-timing subcommand
    analyze_parser = subparsers.add_parser(
        "analyze-timing",
        help="Analyze timing accuracy of lyric videos",
        description="Extract frames at word timestamps and analyze audio to verify timing accuracy. "
        "Useful for debugging timing issues in lyric videos.",
        epilog="""
Examples:
  # Analyze timing for a run
  infomux analyze-timing run-20260120-220733-0cf45b

  # Extract more sample frames
  infomux analyze-timing --frames 20 run-20260120-220733-0cf45b

  # Include audio energy analysis
  infomux analyze-timing --audio-analysis run-20260120-220733-0cf45b
""",
    )
    analyze_timing_cmd.configure_parser(analyze_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the infomux CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = create_parser()
    args: Namespace = parser.parse_args(argv)

    # Configure logging based on verbosity
    log_level = "INFO"
    if args.quiet:
        log_level = "ERROR"
    elif args.verbose >= 2:
        log_level = "DEBUG"
    elif args.verbose == 1:
        log_level = "DEBUG"

    configure_logging(level=log_level)

    logger.debug("infomux %s starting", __version__)
    logger.debug("args: %s", args)

    # Dispatch to subcommand
    try:
        if args.command == "run":
            return run_cmd.execute(args)
        elif args.command == "inspect":
            return inspect_cmd.execute(args)
        elif args.command == "resume":
            return resume_cmd.execute(args)
        elif args.command == "stream":
            return stream_cmd.execute(args)
        elif args.command == "cleanup":
            return cleanup_cmd.execute(args)
        elif args.command == "analyze-timing":
            return analyze_timing_cmd.execute(args)
        else:
            # This shouldn't happen due to required=True on subparsers
            parser.print_help(sys.stderr)
            return 1
    except KeyboardInterrupt:
        logger.info("interrupted by user")
        return 130
    except Exception as e:
        logger.exception("unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
