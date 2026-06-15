"""
Tests for the CLI module.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from infomux.cli import create_parser, main
from infomux.job import InputFile, JobEnvelope, JobStatus, StepRecord
from infomux.storage import list_runs, load_job, save_job


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
        assert args.input == str(test_file)

    def test_run_with_steps(self, tmp_path) -> None:
        """run command accepts --steps."""
        parser = create_parser()
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        args = parser.parse_args(["run", "--steps", "a,b,c", str(test_file)])

        assert args.steps == "a,b,c"

    def test_run_openai_flags(self, tmp_path) -> None:
        """run command accepts OpenAI override flags."""
        parser = create_parser()
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        args = parser.parse_args(
            [
                "run",
                "--openai-model",
                "gpt-4o-mini",
                "--openai-base-url",
                "https://api.openai.com/v1",
                str(test_file),
            ]
        )

        assert args.openai_model == "gpt-4o-mini"
        assert args.openai_base_url == "https://api.openai.com/v1"

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

    def test_cache_command(self) -> None:
        """cache command parses correctly."""
        parser = create_parser()
        args = parser.parse_args(["cache", "external", "status", "--json"])

        assert args.command == "cache"
        assert args.json is True

    def test_watch_command(self, tmp_path) -> None:
        """watch command parses directory and pipeline options."""
        parser = create_parser()
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        args = parser.parse_args(
            [
                "watch",
                str(inbox),
                "--pipeline",
                "transcribe",
                "--glob",
                "*.mp4",
                "--once",
            ]
        )

        assert args.command == "watch"
        assert args.target == str(inbox)
        assert args.pipeline == "transcribe"
        assert args.glob == "*.mp4"
        assert args.once is True

    def test_watch_serve_command(self) -> None:
        """watch serve parses as serve target."""
        parser = create_parser()
        args = parser.parse_args(["watch", "serve", "--once"])
        assert args.command == "watch"
        assert args.target == "serve"
        assert args.once is True

    def test_main_watch_dispatch(self, tmp_path) -> None:
        """main dispatches watch command."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        with patch("infomux.commands.watch.execute", return_value=0) as mock_execute:
            exit_code = main(["watch", str(inbox), "--once"])

        assert exit_code == 0
        assert mock_execute.called

    def test_main_cache_dispatch(self) -> None:
        """main dispatches cache command."""
        with patch("infomux.commands.cache.execute", return_value=0) as mock_execute:
            exit_code = main(["cache", "external", "status"])
        assert exit_code == 0
        assert mock_execute.called

    def test_resume_openai_flags(self) -> None:
        """resume command accepts OpenAI override flags."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "resume",
                "--openai-model",
                "gpt-4.1-mini",
                "--openai-base-url",
                "https://api.openai.com/v1",
                "run-123",
            ]
        )

        assert args.openai_model == "gpt-4.1-mini"
        assert args.openai_base_url == "https://api.openai.com/v1"

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

    def test_main_loads_dotenv_from_cwd(self, tmp_path, monkeypatch) -> None:
        """CLI startup loads .env values before command execution."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("INFOMUX_TEST_DOTENV", raising=False)
        (tmp_path / ".env").write_text("INFOMUX_TEST_DOTENV=loaded\n")

        with patch("infomux.commands.inspect.execute", return_value=0):
            exit_code = main(["inspect", "--list"])

        assert exit_code == 0
        assert "INFOMUX_TEST_DOTENV" in os.environ

    def test_main_missing_command_prints_recovery_tips(self, capsys) -> None:
        """Missing command returns parse error with targeted tips."""
        exit_code = main([])

        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Try one of these:" in captured.err
        assert "infomux run input.mp4" in captured.err
        assert "infomux --help" in captured.err

    def test_main_cache_missing_domain_prints_recovery_tips(self, capsys) -> None:
        """Missing cache domain/action returns parse error with tips."""
        exit_code = main(["cache"])

        assert exit_code == 2
        captured = capsys.readouterr()
        assert "infomux cache external status" in captured.err
        assert "infomux cache external list" in captured.err

    def test_run_keyboard_interrupt_marks_job_interrupted(
        self, tmp_path, monkeypatch
    ) -> None:
        """Interrupted runs persist as interrupted in job.json."""
        monkeypatch.setenv("INFOMUX_DATA_DIR", str(tmp_path))
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video")

        class DummyTools:
            def validate(self) -> list[str]:
                return []

        with (
            patch("infomux.commands.run.get_tool_paths", return_value=DummyTools()),
            patch(
                "infomux.commands.run.run_pipeline",
                side_effect=KeyboardInterrupt,
            ),
        ):
            exit_code = main(["run", str(test_file)])

        assert exit_code == 130
        runs = list_runs()
        assert len(runs) == 1
        job = load_job(runs[0])
        assert job.status == JobStatus.INTERRUPTED.value
        assert job.error == "interrupted by user"

    def test_resume_keyboard_interrupt_marks_job_interrupted(
        self, tmp_path, monkeypatch
    ) -> None:
        """Interrupted resume persists interrupted status."""
        monkeypatch.setenv("INFOMUX_DATA_DIR", str(tmp_path))
        input_path = tmp_path / "resume-input.mp4"
        input_path.write_bytes(b"fake video")
        input_file = InputFile.from_path(input_path)
        job = JobEnvelope.create(input_file=input_file)
        job.config["pipeline"] = "transcribe"
        job.update_status(JobStatus.FAILED, "prior failure")
        job.steps = [StepRecord(name="extract_audio", status="completed")]
        save_job(job)

        class DummyTools:
            def validate(self) -> list[str]:
                return []

        with (
            patch("infomux.commands.resume.get_tool_paths", return_value=DummyTools()),
            patch(
                "infomux.commands.resume.run_pipeline",
                side_effect=KeyboardInterrupt,
            ),
        ):
            exit_code = main(["resume", job.id])

        assert exit_code == 130
        resumed_job = load_job(job.id)
        assert resumed_job.status == JobStatus.INTERRUPTED.value
        assert resumed_job.error == "interrupted by user"
