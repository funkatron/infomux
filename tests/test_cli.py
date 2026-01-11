"""
Tests for the CLI module.
"""

from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import patch

import pytest

from infomux.cli import create_parser, main


class TestParser:
    """Tests for argument parsing."""

    def test_version(self) -> None:
        """--version flag works."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_run_command(self, tmp_path) -> None:
        """run command parses correctly."""
        parser = create_parser()
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        args = parser.parse_args(["run", str(test_file)])

        assert args.command == "run"
        assert args.input == test_file

    def test_run_with_steps(self, tmp_path) -> None:
        """run command accepts --steps."""
        parser = create_parser()
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        args = parser.parse_args(["run", "--steps", "a,b,c", str(test_file)])

        assert args.steps == "a,b,c"

    def test_inspect_command(self) -> None:
        """inspect command parses correctly."""
        parser = create_parser()
        args = parser.parse_args(["inspect", "run-123"])

        assert args.command == "inspect"
        assert args.run_id == "run-123"

    def test_inspect_list(self) -> None:
        """inspect --list parses correctly."""
        parser = create_parser()
        args = parser.parse_args(["inspect", "--list"])

        assert args.command == "inspect"
        assert args.list_runs is True

    def test_resume_command(self) -> None:
        """resume command parses correctly."""
        parser = create_parser()
        args = parser.parse_args(["resume", "run-123"])

        assert args.command == "resume"
        assert args.run_id == "run-123"

    def test_no_command_fails(self) -> None:
        """Missing command fails."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestMain:
    """Tests for the main entry point."""

    def test_run_nonexistent_file(self, tmp_path) -> None:
        """run fails for non-existent input."""
        exit_code = main(["run", str(tmp_path / "nonexistent.mp4")])
        assert exit_code == 1

    def test_run_dry_run(self, tmp_path, capsys) -> None:
        """run --dry-run outputs job envelope without executing."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video")

        exit_code = main(["run", "--dry-run", str(test_file)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert '"id":' in captured.out
        assert '"status": "pending"' in captured.out

    def test_inspect_list_empty(self, tmp_path, monkeypatch) -> None:
        """inspect --list works with no runs."""
        # Use temp dir as data dir
        monkeypatch.setenv("INFOMUX_DATA_DIR", str(tmp_path))

        exit_code = main(["inspect", "--list"])

        assert exit_code == 0
