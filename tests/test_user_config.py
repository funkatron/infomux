"""
Tests for user config loading (TOML).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infomux.user_config import (
    default_config_path,
    load_user_config,
    resolve_config_path,
)


class TestConfigPaths:
    def test_default_config_path(self) -> None:
        path = default_config_path()
        assert path.name == "config.toml"
        assert path.parent.name == "infomux"

    def test_resolve_explicit_path(self, tmp_path: Path) -> None:
        config = tmp_path / "my.toml"
        assert resolve_config_path(config) == config


class TestLoadUserConfig:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        config = load_user_config(tmp_path / "missing.toml")
        assert config.path is None
        assert config.watches == []

    def test_loads_defaults_and_watches(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[defaults]
pipeline = "transcribe"
model = "llama3.2:3b"

[[watch]]
directory = "~/Inbox"
glob = "*.m4a"

[[watch]]
directory = "/tmp/downloads"
pipeline = "summarize"
debounce = 5.0
recursive = false
""".strip()
            + "\n",
            encoding="utf-8",
        )

        config = load_user_config(config_path)

        assert config.path == config_path
        assert config.defaults.pipeline == "transcribe"
        assert config.defaults.model == "llama3.2:3b"
        assert len(config.watches) == 2

        inbox = config.watches[0]
        assert inbox.directory == Path("~/Inbox").expanduser()
        assert inbox.glob == "*.m4a"
        assert inbox.pipeline.pipeline == "transcribe"
        assert inbox.pipeline.model == "llama3.2:3b"

        downloads = config.watches[1]
        assert downloads.directory == Path("/tmp/downloads")
        assert downloads.pipeline.pipeline == "summarize"
        assert downloads.debounce == 5.0
        assert downloads.recursive is False

    def test_watch_inherits_defaults(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[defaults]
content_type_hint = "meeting"

[[watch]]
directory = "~/Calls"
pipeline = "summarize"
""".strip()
            + "\n",
            encoding="utf-8",
        )

        config = load_user_config(config_path)
        watch = config.watches[0]
        assert watch.pipeline.pipeline == "summarize"
        assert watch.pipeline.content_type_hint == "meeting"

    def test_invalid_toml_raises(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.toml"
        config_path.write_text("[[watch]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid TOML"):
            load_user_config(config_path)

    def test_watch_requires_directory(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text("[[watch]]\npipeline = 'transcribe'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required 'directory'"):
            load_user_config(config_path)

    def test_serve_section(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[serve]
fswatch = "/opt/homebrew/bin/fswatch"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        config = load_user_config(config_path)
        assert config.serve.fswatch == Path("/opt/homebrew/bin/fswatch")
