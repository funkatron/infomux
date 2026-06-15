"""
Tests for directory watching helpers.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from infomux.watcher import (
    DirectoryWatcher,
    build_fswatch_command,
    find_fswatch,
    is_processed,
    load_registry,
    mark_processed,
    should_skip_path,
)


class TestShouldSkipPath:
    def test_skips_hidden_and_temp_files(self, tmp_path: Path) -> None:
        assert should_skip_path(tmp_path / ".hidden.mp4")
        assert should_skip_path(tmp_path / "clip.part")
        assert should_skip_path(tmp_path / "clip.mp4.tmp")
        assert should_skip_path(tmp_path / ".infomux-watch.json")

    def test_allows_normal_media(self, tmp_path: Path) -> None:
        assert not should_skip_path(tmp_path / "episode.mp3")


class TestRegistry:
    def test_mark_and_detect_processed(self, tmp_path: Path) -> None:
        media = tmp_path / "clip.mp4"
        media.write_bytes(b"video")
        identity = (media.stat().st_mtime_ns, media.stat().st_size)

        registry = load_registry(tmp_path / ".infomux-watch.json")
        assert not is_processed(registry, media, identity)

        mark_processed(registry, media, identity)
        assert is_processed(registry, media, identity)


class TestFswatchHelpers:
    def test_build_fswatch_command_recursive(self, tmp_path: Path) -> None:
        cmd = build_fswatch_command(
            Path("/usr/local/bin/fswatch"),
            tmp_path,
            recursive=True,
            latency_seconds=0.5,
        )
        assert cmd == ["/usr/local/bin/fswatch", "-l", "0.5", "-r", str(tmp_path)]

    def test_build_fswatch_command_non_recursive(self, tmp_path: Path) -> None:
        cmd = build_fswatch_command(
            Path("/usr/local/bin/fswatch"),
            tmp_path,
            recursive=False,
            latency_seconds=2.0,
        )
        assert cmd == ["/usr/local/bin/fswatch", "-l", "2.0", str(tmp_path)]

    def test_find_fswatch_explicit(self, tmp_path: Path) -> None:
        binary = tmp_path / "fswatch"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        assert find_fswatch(binary) == binary.resolve()


class TestDirectoryWatcher:
    def test_run_once_processes_new_files(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        clip = inbox / "note.m4a"
        clip.write_bytes(b"audio")

        processed: list[Path] = []

        def process(path: Path) -> int:
            processed.append(path)
            return 0

        watcher = DirectoryWatcher(
            directory=inbox,
            process=process,
            glob_pattern="*.m4a",
        )
        exit_code = watcher.run_once()

        assert exit_code == 0
        assert processed == [clip]

    def test_run_once_skips_registry_entries(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        clip = inbox / "note.m4a"
        clip.write_bytes(b"audio")

        calls = 0

        def process(path: Path) -> int:
            nonlocal calls
            calls += 1
            return 0

        watcher = DirectoryWatcher(
            directory=inbox,
            process=process,
            glob_pattern="*.m4a",
        )
        assert watcher.run_once() == 0
        assert calls == 1
        assert watcher.run_once() == 0
        assert calls == 1

    def test_handle_event_schedules_processing(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        clip = inbox / "note.m4a"
        clip.write_bytes(b"audio")

        processed: list[Path] = []

        class ImmediateTimer:
            def __init__(self, _delay: float, fn) -> None:
                self.fn = fn

            def start(self) -> None:
                self.fn()

            def cancel(self) -> None:
                pass

        watcher = DirectoryWatcher(
            directory=inbox,
            process=lambda path: processed.append(path) or 0,
            glob_pattern="*.m4a",
            debounce_seconds=2.0,
        )

        with patch("infomux.watcher.threading.Timer", ImmediateTimer):
            watcher.handle_event_path(str(clip))

        assert processed == [clip]

    def test_run_forever_requires_fswatch(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        watcher = DirectoryWatcher(
            directory=inbox,
            process=lambda _path: 0,
        )

        with patch("infomux.watcher.find_fswatch", return_value=None):
            assert watcher.run_forever() == 1

    def test_run_forever_reads_fswatch_events(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        clip = inbox / "note.m4a"
        clip.write_bytes(b"audio")

        processed: list[Path] = []

        class FakeStdout:
            def __init__(self, lines: list[str]) -> None:
                self._lines = iter(lines)

            def __iter__(self):
                return self

            def __next__(self) -> str:
                return next(self._lines)

        fake_proc = MagicMock()
        fake_proc.stdout = FakeStdout([f"{clip}\n"])
        fake_proc.stderr = None
        fake_proc.poll.return_value = 0
        fake_proc.returncode = 0

        watcher = DirectoryWatcher(
            directory=inbox,
            process=lambda path: processed.append(path) or 0,
            glob_pattern="*.m4a",
            debounce_seconds=0.0,
            fswatch_path=Path("/usr/local/bin/fswatch"),
        )

        class ImmediateTimer:
            def __init__(self, _delay: float, fn) -> None:
                self.fn = fn

            def start(self) -> None:
                self.fn()

            def cancel(self) -> None:
                pass

        with (
            patch("infomux.watcher.find_fswatch", return_value=Path("/usr/local/bin/fswatch")),
            patch("infomux.watcher.subprocess.Popen", return_value=fake_proc),
            patch("infomux.watcher.threading.Timer", ImmediateTimer),
        ):
            assert watcher.run_forever() == 0

        assert processed == [clip]

    def test_run_forever_handles_keyboard_interrupt(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        def raise_interrupt(_cmd, **_kwargs):
            raise KeyboardInterrupt

        watcher = DirectoryWatcher(
            directory=inbox,
            process=lambda _path: 0,
            fswatch_path=Path("/usr/local/bin/fswatch"),
        )

        with (
            patch("infomux.watcher.find_fswatch", return_value=Path("/usr/local/bin/fswatch")),
            patch("infomux.watcher.subprocess.Popen", side_effect=raise_interrupt),
        ):
            assert watcher.run_forever() == 130
