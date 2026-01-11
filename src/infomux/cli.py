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
  infomux run input.mp4              Run pipeline on a media file
  infomux stream                     Record and transcribe live
  infomux stream --device 0          Record from specific device
  infomux inspect abc123             Inspect a previous run
  infomux resume abc123              Resume an interrupted run

Environment:
  INFOMUX_DATA_DIR    Base directory for runs (default: ~/.local/share/infomux)
  INFOMUX_LOG_LEVEL   Log verbosity: DEBUG, INFO, WARN, ERROR (default: INFO)
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
        description="Process a media file through the configured pipeline steps.",
    )
    run_cmd.configure_parser(run_parser)

    # inspect subcommand
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a previous run",
        description="Display details about a completed or in-progress run.",
    )
    inspect_cmd.configure_parser(inspect_parser)

    # resume subcommand
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an interrupted run",
        description="Continue a run that was interrupted or failed.",
    )
    resume_cmd.configure_parser(resume_parser)

    # stream subcommand
    stream_parser = subparsers.add_parser(
        "stream",
        help="Real-time audio capture and transcription",
        description="Record from microphone and transcribe in real-time.",
    )
    stream_cmd.configure_parser(stream_parser)

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
