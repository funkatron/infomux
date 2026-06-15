"""
The 'watch' command: run a pipeline when new files appear in a directory.

Uses fswatch for filesystem events, waits until each file stops changing
(debounce), then runs the same pipeline logic as `infomux run`. Processed files
are tracked in `.infomux-watch.json` in the watch directory.

Usage:
    infomux watch ~/Inbox --pipeline transcribe
    infomux watch serve
    infomux watch ~/Downloads --glob "*.mp4" --pipeline summarize --once
"""

from __future__ import annotations

import threading
from argparse import ArgumentParser, Namespace
from pathlib import Path

from infomux.commands import run as run_cmd
from infomux.log import get_logger
from infomux.user_config import (
    WatchEntry,
    apply_defaults_to_args,
    default_config_path,
    load_user_config,
)
from infomux.watcher import DirectoryWatcher

logger = get_logger(__name__)


def configure_parser(parser: ArgumentParser) -> None:
    """Configure the argument parser for the 'watch' command."""
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Directory to watch, or 'serve' to run all [[watch]] entries from config.toml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Config file (default: {default_config_path()}).",
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


def _build_watcher(
    *,
    directory: Path,
    pipeline_args: Namespace,
    glob_pattern: str,
    recursive: bool,
    debounce_seconds: float,
    registry_path: Path | None,
    fswatch_path: Path | None,
    record_processed: bool,
) -> DirectoryWatcher:
    def process_file(path: Path) -> int:
        run_args = run_cmd.build_run_namespace(pipeline_args, path)
        return run_cmd.execute(run_args)

    return DirectoryWatcher(
        directory=directory,
        process=process_file,
        glob_pattern=glob_pattern,
        recursive=recursive,
        debounce_seconds=debounce_seconds,
        registry_path=registry_path,
        fswatch_path=fswatch_path,
        record_processed=record_processed,
    )


def _validate_directory(directory: Path) -> int | None:
    if not directory.exists():
        logger.error("watch directory not found: %s", directory)
        return 1
    if not directory.is_dir():
        logger.error("watch path is not a directory: %s", directory)
        return 1
    return None


def _execute_single(args: Namespace) -> int:
    """Watch one directory from CLI arguments."""
    directory = Path(args.target).expanduser().resolve()
    error = _validate_directory(directory)
    if error is not None:
        return error

    try:
        config = load_user_config(args.config)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    apply_defaults_to_args(args, config.defaults)

    watcher = _build_watcher(
        directory=directory,
        pipeline_args=args,
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


def _watcher_from_entry(
    entry: WatchEntry,
    *,
    fswatch_path: Path | None,
    cli_dry_run: bool,
) -> DirectoryWatcher | None:
    directory = entry.directory.expanduser().resolve()
    error = _validate_directory(directory)
    if error is not None:
        return None

    pipeline_args = entry.pipeline.to_namespace()
    effective_dry_run = cli_dry_run or entry.pipeline.dry_run
    pipeline_args.dry_run = effective_dry_run

    return _build_watcher(
        directory=directory,
        pipeline_args=pipeline_args,
        glob_pattern=entry.glob,
        recursive=entry.recursive,
        debounce_seconds=entry.debounce,
        registry_path=entry.registry,
        fswatch_path=fswatch_path,
        record_processed=not effective_dry_run,
    )


def _execute_serve(args: Namespace) -> int:
    """Run all [[watch]] entries from config.toml."""
    try:
        config = load_user_config(args.config)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    if not config.watches:
        path = config.path or default_config_path()
        logger.error("no [[watch]] entries in config: %s", path)
        return 1

    fswatch_path = args.fswatch or config.serve.fswatch
    cli_dry_run = bool(args.dry_run)

    watchers: list[DirectoryWatcher] = []
    for entry in config.watches:
        watcher = _watcher_from_entry(entry, fswatch_path=fswatch_path, cli_dry_run=cli_dry_run)
        if watcher is not None:
            watchers.append(watcher)

    if not watchers:
        return 1

    if args.once:
        exit_code = 0
        for watcher in watchers:
            result = watcher.run_once()
            if result != 0:
                exit_code = result
        return exit_code

    logger.info("watch serve: starting %d watches from %s", len(watchers), config.path)
    threads: list[threading.Thread] = []
    results: dict[int, int] = {}

    def run_in_thread(index: int, watcher: DirectoryWatcher) -> None:
        results[index] = watcher.run_forever()

    for index, watcher in enumerate(watchers):
        thread = threading.Thread(
            target=run_in_thread,
            args=(index, watcher),
            name=f"infomux-watch-{index}",
            daemon=True,
        )
        threads.append(thread)
        thread.start()

    exit_code = 0
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logger.info("watch serve stopped")
        for watcher in watchers:
            watcher.stop()
        return 130

    for result in results.values():
        if result != 0:
            exit_code = result
    return exit_code


def execute(args: Namespace) -> int:
    """
    Execute the 'watch' command.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    if args.target is None:
        logger.error("specify a directory to watch or 'serve' (from config.toml)")
        return 1

    if args.target == "serve":
        return _execute_serve(args)

    return _execute_single(args)
