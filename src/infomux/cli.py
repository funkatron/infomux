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
from infomux.commands import audio_recon as audio_recon_cmd
from infomux.commands import cache as cache_cmd
from infomux.commands import cleanup as cleanup_cmd
from infomux.commands import inspect as inspect_cmd
from infomux.commands import resume as resume_cmd
from infomux.commands import run as run_cmd
from infomux.commands import stream as stream_cmd
from infomux.commands import watch as watch_cmd
from infomux.env import load_dotenv
from infomux.log import configure_logging, get_logger

if TYPE_CHECKING:
    from argparse import Namespace

logger = get_logger(__name__)


def _print_parse_tips(argv: list[str]) -> None:
    """
    Print concise recovery tips after argparse errors.

    Tier-1 coverage:
    - Missing top-level command
    - Missing required positional/subcommand for common commands
    """
    known_commands = {
        "run",
        "inspect",
        "resume",
        "stream",
        "cleanup",
        "cache",
        "analyze-timing",
        "audio-recon",
        "watch",
    }
    first_token = argv[0] if argv else None

    # Missing command (e.g., `infomux` or only global flags)
    if not argv or (first_token and first_token.startswith("-")):
        print("\nTry one of these:", file=sys.stderr)
        print("  infomux run input.mp4", file=sys.stderr)
        print("  infomux stream  # default: mic + loopback when available", file=sys.stderr)
        print("  infomux inspect --list", file=sys.stderr)
        print("  infomux --help", file=sys.stderr)
        return

    # Unknown top-level command
    if first_token not in known_commands:
        print("\nUnknown command. Available commands:", file=sys.stderr)
        print(
            "  run, inspect, resume, stream, watch, cleanup, cache, analyze-timing, audio-recon",
            file=sys.stderr,
        )
        print("  infomux --help", file=sys.stderr)
        return

    # Command-specific required positional/subcommand guidance.
    if first_token == "resume" and len(argv) == 1:
        print("\nTry:", file=sys.stderr)
        print("  infomux resume <run-id>", file=sys.stderr)
        print("  infomux inspect --list", file=sys.stderr)
        return

    if first_token == "analyze-timing" and len(argv) == 1:
        print("\nTry:", file=sys.stderr)
        print("  infomux analyze-timing <run-id>", file=sys.stderr)
        print("  infomux inspect --list", file=sys.stderr)
        return

    if first_token == "cache":
        if len(argv) == 1:
            print("\nTry:", file=sys.stderr)
            print("  infomux cache external status", file=sys.stderr)
            print("  infomux cache external list", file=sys.stderr)
            print("  infomux cache external clear --yes", file=sys.stderr)
            return
        if len(argv) == 2 and argv[1] == "external":
            print("\nTry:", file=sys.stderr)
            print("  infomux cache external status", file=sys.stderr)
            print("  infomux cache external path", file=sys.stderr)
            print("  infomux cache external list", file=sys.stderr)
            print("  infomux cache external clear --yes", file=sys.stderr)
            return


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
  infomux stream --prompt
  infomux stream --input 0 --output 1 --silence 5
  infomux stream --pipeline summarize

  # Watch a folder and run a pipeline on new files
  infomux watch ~/Inbox --pipeline transcribe
  infomux watch ~/Downloads --glob "*.mp4" --pipeline summarize --once

  # Quick loopback / system-audio check (auto device selection)
  infomux audio-recon

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

  # Inspect/manage external service cache
  infomux cache external status
  infomux cache external list
  infomux cache external clear --yes

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
        description=(
            "Process a media file (or URL) through the configured pipeline steps. "
            "Supports audio, video, and text files. Automatically detects HTML content "
            "and uses the web-summarize pipeline when appropriate."
        ),
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
        "Useful for recovering from errors or re-running steps with different "
        "settings.",
        epilog="""
Examples:
  # Resume a failed run (continues from where it stopped)
  infomux resume run-20260111-020549-c36c19

  # Re-run transcription step and all following steps
  infomux resume --from-step transcribe run-20260111-020549-c36c19

  # Re-generate summary with different model
  infomux resume --from-step summarize --model qwen2.5:32b-instruct \\
      run-20260111-020549-c36c19

  # Re-summarize with content type hint
  infomux resume --from-step summarize --content-type-hint meeting \\
      run-20260111-020549-c36c19

  # Preview what would be re-run
  infomux resume --dry-run run-20260111-020549-c36c19
""",
    )
    resume_cmd.configure_parser(resume_parser)

    # stream subcommand
    stream_parser = subparsers.add_parser(
        "stream",
        help="Real-time audio capture and transcription",
        description="Record from audio devices and transcribe in real-time. "
        "By default uses the system default input plus a loopback device when "
        "available (for mixed mic + system audio). Use --list-devices for IDs; "
        "use --input/--output to override. "
        "Supports stop conditions: duration, silence detection, or stop phrase. "
        "Can run additional pipelines (like summarize) after recording completes.",
        epilog="""
Examples:
  # Default capture (default input + default loopback when available)
  infomux stream

  # Interactive device selection with live meters
  infomux stream --prompt

  # Explicit input and output device IDs (from --list-devices)
  infomux stream --input 1 --output 2

  # Legacy: single microphone only, no loopback (older CLI behavior)
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
  infomux stream --input 1 --silence 10 --pipeline summarize

  # List INPUTS and OUTPUTS with device IDs
  infomux stream --list-devices
""",
    )
    stream_cmd.configure_parser(stream_parser)

    # cleanup subcommand
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Remove orphaned or unwanted runs",
        description="Clean up the runs directory by removing orphaned runs, stuck runs, or runs matching specific criteria. "
        "Cleanup previews matches by default; pass --force to actually delete them.",
        epilog="""
Examples:
  # Preview orphaned runs (default behavior)
  infomux cleanup --orphaned

  # Delete orphaned runs (no valid job.json)
  infomux cleanup --force --orphaned

  # Preview stuck runs (status: running)
  infomux cleanup --status running

  # Delete runs older than 30 days
  infomux cleanup --force --older-than 30d

  # Preview failed runs older than 7 days (with safety check)
  infomux cleanup --status failed --older-than 7d --min-age 1d

  # Combine filters: preview orphaned and stuck runs
  infomux cleanup --orphaned --status running

Time specifications:
  Use 'd' for days, 'w' for weeks, 'm' for months
  Examples: 30d, 2w, 1m
""",
    )
    cleanup_cmd.configure_parser(cleanup_parser)

    # cache subcommand
    cache_parser = subparsers.add_parser(
        "cache",
        help="Inspect and manage local caches",
        description="Manage caches by domain. "
        "Example: external service response caches.",
        epilog="""
Examples:
  # Show external cache status (provider, path, file count, bytes)
  infomux cache external status

  # Print cache path only
  infomux cache external path

  # List cached files
  infomux cache external list

  # Clear cache files (with confirmation)
  infomux cache external clear

  # Clear cache files without prompt
  infomux cache external clear --yes
""",
    )
    cache_cmd.configure_parser(cache_parser)

    # analyze-timing subcommand
    analyze_parser = subparsers.add_parser(
        "analyze-timing",
        help="Analyze timing accuracy of lyric videos",
        description="Extract frames at word timestamps and analyze audio to "
        "verify timing accuracy. "
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

    audio_recon_parser = subparsers.add_parser(
        "audio-recon",
        help="Quick check that system audio reaches a recordable device",
        description=(
            "Records a short sample from an automatically chosen loopback-capable "
            "device (override with INFOMUX_RECON_CAPTURE or --output-name). "
            "Measures peak level; exits 0 if not silent, 2 if silent, 1 on error. "
            "Optional --switch-output uses SwitchAudioSource (brew install switchaudio-osx)."
        ),
        epilog="""
Examples:
  infomux audio-recon
  infomux audio-recon --duration 5 --json
  infomux audio-recon --switch-output "infomux-capture" --sleep-after-switch 2
  infomux audio-recon --output-name "infomux-aggregate-device" --play
  infomux audio-recon --check-only
""",
    )
    audio_recon_cmd.configure_parser(audio_recon_parser)

    watch_parser = subparsers.add_parser(
        "watch",
        help="Watch a directory and run a pipeline on new files",
        description=(
            "Watch a directory with fswatch for new media files. When a file stops "
            "changing (debounce), run the same pipeline as `infomux run`. Processed "
            "files are tracked in .infomux-watch.json so restarts skip completed work."
        ),
        epilog="""
Examples:
  # Transcribe anything dropped into ~/Inbox
  infomux watch ~/Inbox --pipeline transcribe

  # Summarize new MP4s only
  infomux watch ~/Downloads --glob "*.mp4" --pipeline summarize

  # Drain existing files once, then exit
  infomux watch ~/Inbox --pipeline transcribe --once

  # Same pipeline flags as run (model, content-type hint, etc.)
  infomux watch ~/Inbox --pipeline summarize --model qwen2.5:32b-instruct
""",
    )
    watch_cmd.configure_parser(watch_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the infomux CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Load .env config before parsing args and configuring logging.
    load_dotenv()

    parser = create_parser()
    argv_list = argv if argv is not None else sys.argv[1:]

    try:
        args: Namespace = parser.parse_args(argv_list)
    except SystemExit as e:
        # argparse uses exit code 2 for parse errors.
        code = e.code if isinstance(e.code, int) else 1
        if code == 2:
            _print_parse_tips(argv_list)
        return code

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
        elif args.command == "cache":
            return cache_cmd.execute(args)
        elif args.command == "analyze-timing":
            return analyze_timing_cmd.execute(args)
        elif args.command == "audio-recon":
            return audio_recon_cmd.execute(args)
        elif args.command == "watch":
            return watch_cmd.execute(args)
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
