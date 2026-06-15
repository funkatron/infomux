"""
Directory watching helpers for infomux watch.

Uses fswatch (https://github.com/emcrisostomo/fswatch) for filesystem events.
After a file stops changing (debounce timer), runs the pipeline callback.
Tracks processed files in a JSON registry so restarts skip work already done.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infomux.log import get_logger

logger = get_logger(__name__)

REGISTRY_FILENAME = ".infomux-watch.json"
REGISTRY_VERSION = 1
ENV_FSWATCH_PATH = "INFOMUX_FSWATCH_PATH"

SKIP_SUFFIXES = (".part", ".tmp", ".crdownload", ".download", ".partial")
SKIP_BASENAMES = {REGISTRY_FILENAME}


def find_fswatch(explicit: Path | str | None = None) -> Path | None:
    """Locate the fswatch binary (explicit path, env, or PATH)."""
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
        return None

    env_path = os.environ.get(ENV_FSWATCH_PATH)
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    found = shutil.which("fswatch")
    return Path(found).resolve() if found else None


def should_skip_path(path: Path) -> bool:
    """Return True for hidden, temp, or registry files."""
    name = path.name
    if name.startswith("."):
        return True
    if name in SKIP_BASENAMES:
        return True
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in SKIP_SUFFIXES)


def file_identity(path: Path) -> tuple[int, int]:
    """Return (mtime_ns, size_bytes) for change detection."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def load_registry(registry_path: Path) -> dict[str, Any]:
    """Load the watch registry or return an empty structure."""
    if not registry_path.exists():
        return {"version": REGISTRY_VERSION, "processed": {}}
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read watch registry %s: %s", registry_path, exc)
        return {"version": REGISTRY_VERSION, "processed": {}}
    if not isinstance(data, dict):
        return {"version": REGISTRY_VERSION, "processed": {}}
    processed = data.get("processed")
    if not isinstance(processed, dict):
        processed = {}
    return {"version": REGISTRY_VERSION, "processed": processed}


def save_registry(registry_path: Path, data: dict[str, Any]) -> None:
    """Persist the watch registry."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": REGISTRY_VERSION, "processed": data.get("processed", {})}
    registry_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def is_processed(registry: dict[str, Any], path: Path, identity: tuple[int, int]) -> bool:
    """Return True if this exact file version was already handled."""
    key = str(path.resolve())
    entry = registry.get("processed", {}).get(key)
    if not isinstance(entry, dict):
        return False
    return entry.get("mtime_ns") == identity[0] and entry.get("size") == identity[1]


def mark_processed(registry: dict[str, Any], path: Path, identity: tuple[int, int]) -> None:
    """Record a successfully processed file version."""
    key = str(path.resolve())
    registry.setdefault("processed", {})[key] = {
        "mtime_ns": identity[0],
        "size": identity[1],
    }


def build_fswatch_command(
    fswatch: Path,
    directory: Path,
    *,
    recursive: bool,
    latency_seconds: float,
) -> list[str]:
    """Build an fswatch argv list for directory notifications."""
    cmd = [
        str(fswatch),
        "-l",
        str(max(latency_seconds, 0.01)),
    ]
    if recursive:
        cmd.append("-r")
    cmd.append(str(directory))
    return cmd


@dataclass
class DirectoryWatcher:
    """
    Watch a directory via fswatch and run a callback on new, stable files.

    Files are processed one at a time. Failures are logged but not recorded in
    the registry, so a later event can retry.
    """

    directory: Path
    process: Callable[[Path], int]
    glob_pattern: str = "*"
    recursive: bool = True
    debounce_seconds: float = 2.0
    registry_path: Path | None = None
    fswatch_path: Path | None = None
    _registry: dict[str, Any] = field(default_factory=dict, init=False)
    _timers: dict[Path, threading.Timer] = field(default_factory=dict, init=False)
    _process_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self.directory = self.directory.expanduser().resolve()
        if self.registry_path is None:
            self.registry_path = self.directory / REGISTRY_FILENAME
        else:
            self.registry_path = self.registry_path.expanduser().resolve()
        self._registry = load_registry(self.registry_path)

    def path_matches(self, path: Path) -> bool:
        """Return True if path is a watchable file under directory matching glob."""
        if should_skip_path(path):
            return False
        try:
            path.resolve().relative_to(self.directory)
        except ValueError:
            return False
        if not self.recursive and path.parent != self.directory:
            return False
        if not path.is_file():
            return False
        return path.match(self.glob_pattern)

    def iter_matching_files(self) -> list[Path]:
        """Return regular files under directory matching glob_pattern."""
        if not self.directory.exists():
            return []
        if not self.directory.is_dir():
            return []

        if self.recursive:
            paths = self.directory.rglob(self.glob_pattern)
        else:
            paths = self.directory.glob(self.glob_pattern)

        files: list[Path] = []
        for path in paths:
            if self.path_matches(path):
                files.append(path)
        return sorted(files)

    def _cancel_timers(self) -> None:
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()

    def _schedule_processing(self, path: Path) -> None:
        """Reset debounce timer for a file; process when it stays stable."""
        path = path.resolve()
        try:
            identity = file_identity(path)
        except OSError as exc:
            logger.debug("skipping unreadable file %s: %s", path, exc)
            return

        if is_processed(self._registry, path, identity):
            return

        def fire() -> None:
            self._timers.pop(path, None)
            if not self.path_matches(path):
                return
            try:
                current = file_identity(path)
            except OSError:
                return
            if current != identity:
                self._schedule_processing(path)
                return
            if is_processed(self._registry, path, current):
                return
            with self._process_lock:
                self._process_file(path)

        existing = self._timers.get(path)
        if existing is not None:
            existing.cancel()
        timer = threading.Timer(self.debounce_seconds, fire)
        timer.daemon = True
        self._timers[path] = timer
        timer.start()

    def handle_event_path(self, raw_path: str) -> None:
        """Handle one path emitted by fswatch."""
        path = Path(raw_path.strip())
        if not path.is_absolute():
            path = (self.directory / path).resolve()
        else:
            path = path.resolve()
        if not self.path_matches(path):
            return
        logger.debug("watch: event for %s", path)
        self._schedule_processing(path)

    def _process_file(self, path: Path) -> int:
        """Run the callback and update the registry on success."""
        try:
            identity = file_identity(path)
        except OSError as exc:
            logger.error("file disappeared before processing: %s (%s)", path, exc)
            return 1

        logger.info("watch: processing %s", path)
        exit_code = self.process(path)
        if exit_code == 0:
            mark_processed(self._registry, path, identity)
            save_registry(self.registry_path, self._registry)
            logger.info("watch: completed %s", path)
        else:
            logger.error("watch: pipeline failed for %s (exit %d)", path, exit_code)
        return exit_code

    def run_once(self) -> int:
        """Process all unprocessed files once (no debounce wait)."""
        exit_code = 0
        for path in self.iter_matching_files():
            try:
                identity = file_identity(path)
            except OSError:
                continue
            if is_processed(self._registry, path, identity):
                continue
            result = self._process_file(path)
            if result != 0:
                exit_code = result
        return exit_code

    def run_forever(self) -> int:
        """Watch with fswatch until interrupted."""
        fswatch = find_fswatch(self.fswatch_path)
        if fswatch is None:
            logger.error(
                "fswatch not found. Install via: brew install fswatch "
                "(or set %s)",
                ENV_FSWATCH_PATH,
            )
            return 1

        cmd = build_fswatch_command(
            fswatch,
            self.directory,
            recursive=self.recursive,
            latency_seconds=min(self.debounce_seconds, 1.0),
        )
        logger.info(
            "watching %s with fswatch (glob=%r, debounce=%.1fs)",
            self.directory,
            self.glob_pattern,
            self.debounce_seconds,
        )
        logger.debug("fswatch command: %s", " ".join(cmd))

        exit_code = 0
        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.strip():
                    self.handle_event_path(line)
                if proc.poll() is not None:
                    break
            if proc.returncode not in (0, None):
                stderr = proc.stderr.read() if proc.stderr is not None else ""
                logger.error("fswatch exited with code %s: %s", proc.returncode, stderr.strip())
                exit_code = proc.returncode or 1
        except KeyboardInterrupt:
            logger.info("watch stopped")
            exit_code = 130 if exit_code == 0 else exit_code
        finally:
            self._cancel_timers()
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        return exit_code
