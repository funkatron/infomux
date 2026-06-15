"""
Tests for watch serve mode.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from infomux.commands.watch import execute


class TestWatchServe:
    def test_serve_run_once(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        clip = inbox / "note.m4a"
        clip.write_bytes(b"audio")

        config_path = tmp_path / "config.toml"
        config_path.write_text(
            f"""
[[watch]]
directory = "{inbox}"
pipeline = "transcribe"
glob = "*.m4a"
""".strip()
            + "\n",
            encoding="utf-8",
        )

        args = Namespace(
            target="serve",
            config=config_path,
            fswatch=None,
            dry_run=False,
            once=True,
        )

        with patch("infomux.commands.run.execute", return_value=0) as mock_run:
            exit_code = execute(args)

        assert exit_code == 0
        mock_run.assert_called_once()
        run_args = mock_run.call_args[0][0]
        assert str(clip) == run_args.input
        assert run_args.pipeline == "transcribe"

    def test_serve_without_watches_fails(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text("[defaults]\npipeline = 'transcribe'\n", encoding="utf-8")

        args = Namespace(
            target="serve",
            config=config_path,
            fswatch=None,
            dry_run=False,
            once=True,
        )
        assert execute(args) == 1
