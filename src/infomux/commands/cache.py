"""
The 'cache' command: inspect and manage local caches.

Uses a hierarchical structure so additional cache domains can be added cleanly:
    infomux cache external <status|path|list|clear>
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace

from infomux.cache import (
    clear_openai_cache,
    get_openai_cache_dir,
    get_openai_cache_stats,
    list_openai_cache_files,
)
from infomux.log import get_logger

logger = get_logger(__name__)


def _add_provider_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai"],
        help="External service cache provider (default: openai).",
    )


def configure_parser(parser: ArgumentParser) -> None:
    """Configure argument parser for the 'cache' command."""
    domains = parser.add_subparsers(
        dest="cache_domain", required=True, metavar="<domain>"
    )

    # external domain
    external = domains.add_parser(
        "external",
        help="Manage external-service response caches",
        description="Inspect and manage external service caches (currently OpenAI).",
    )
    actions = external.add_subparsers(
        dest="cache_action", required=True, metavar="<action>"
    )

    status = actions.add_parser("status", help="Show cache status summary")
    _add_provider_arg(status)
    status.add_argument("--json", action="store_true", help="Output status as JSON.")

    path = actions.add_parser("path", help="Print cache directory path")
    _add_provider_arg(path)

    list_cmd = actions.add_parser("list", help="List cache files")
    _add_provider_arg(list_cmd)
    list_cmd.add_argument(
        "--json", action="store_true", help="Output file list as JSON."
    )

    clear = actions.add_parser("clear", help="Delete cache files")
    _add_provider_arg(clear)
    clear.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )


def _execute_external(args: Namespace) -> int:
    """Execute cache commands for external services."""
    if args.provider != "openai":
        logger.error("unsupported provider: %s", args.provider)
        return 1

    cache_dir = get_openai_cache_dir()
    action = args.cache_action

    if action == "path":
        print(str(cache_dir), file=sys.stdout)
        return 0

    if action == "clear":
        if not getattr(args, "yes", False):
            answer = (
                input(
                    f"Delete all cache files for provider '{args.provider}' in {cache_dir}? [y/N]: "
                )
                .strip()
                .lower()
            )
            if answer not in {"y", "yes"}:
                print("Aborted.")
                return 0
        deleted = clear_openai_cache()
        print(f"Deleted {deleted} cache file(s).", file=sys.stdout)
        return 0

    if action == "list":
        files = list_openai_cache_files()
        if getattr(args, "json", False):
            print(json.dumps([str(p) for p in files], indent=2), file=sys.stdout)
            return 0
        if not files:
            print("No cache files found.", file=sys.stdout)
            return 0
        for p in files:
            print(str(p), file=sys.stdout)
        return 0

    # status
    file_count, total_bytes = get_openai_cache_stats()
    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "provider": args.provider,
                    "path": str(cache_dir),
                    "file_count": file_count,
                    "total_bytes": total_bytes,
                },
                indent=2,
            ),
            file=sys.stdout,
        )
        return 0

    print(f"Provider: {args.provider}", file=sys.stdout)
    print(f"Path: {cache_dir}", file=sys.stdout)
    print(f"Files: {file_count}", file=sys.stdout)
    print(f"Size: {total_bytes} bytes", file=sys.stdout)
    return 0


def execute(args: Namespace) -> int:
    """Execute the 'cache' command."""
    if args.cache_domain == "external":
        return _execute_external(args)
    logger.error("unsupported cache domain: %s", args.cache_domain)
    return 1
