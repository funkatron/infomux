"""
The 'watch' command: run a pipeline when new files appear in a directory.

Uses fswatch for filesystem events, waits until each file stops changing
(debounce), then runs the same pipeline logic as `infomux run`. Processed files
are tracked in `.infomux-watch.json` in the watch directory.

Usage:
    infomux watch ~/Inbox --pipeline transcribe
    infomux watch ~/Downloads --glob "*.mp4" --pipeline summarize --once
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from infomux.commands import run as run_cmd
from infomux.log import get_logger
from infomux.watcher import DirectoryWatcher

logger = get_logger(__name__)


def configure_parser(parser: ArgumentParser) -> None:
    """Configure the argument parser for the 'watch' command."""
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to watch for new media files.",
    )
    run_cmd.add_pipeline_arguments(parser)
    parser.add_argument(
        "--glob",
        type=str,
        default="*",
        help="Filename glob to match (default: *). Example: '*.mp4' or '*.m4a'.",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Wait until file size/mtime is unchanged for this long before running "
        "(default: 2.0). Skipped with --once.",
    )
    parser.add_argument(
        "--fswatch",
        type=Path,
        default=None,
        help="Path to fswatch binary (default: search PATH or INFOMUX_FSWATCH_PATH).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to processed-file registry (default: <directory>/.infomux-watch.json).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process existing unhandled files and exit (no continuous watch).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only watch the top-level directory, not subfolders.",
    )


def execute(args: Namespace) -> int:
    """
    Execute the 'watch' command.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    directory = args.directory.expanduser().resolve()
    if not directory.exists():
        logger.error("watch directory not found: %s", directory)
        return 1
    if not directory.is_dir():
        logger.error("watch path is not a directory: %s", directory)
        return 1

    def process_file(path: Path) -> int:
        run_args = run_cmd.build_run_namespace(args, path)
        return run_cmd.execute(run_args)

    watcher = DirectoryWatcher(
        directory=directory,
        process=process_file,
        glob_pattern=args.glob,
        recursive=not args.no_recursive,
        debounce_seconds=args.debounce,
        registry_path=args.registry,
        fswatch_path=args.fswatch,
        record_processed=not args.dry_run,
    )

    if args.once:
        return watcher.run_once()
    return watcher.run_forever()
