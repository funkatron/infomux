"""
Tests for dotenv loading.
"""

from __future__ import annotations

from pathlib import Path

from infomux.env import load_dotenv


def test_load_dotenv_loads_values(tmp_path: Path, monkeypatch) -> None:
    """Loads key/value pairs from .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "INFOMUX_TEST_A=alpha",
                "INFOMUX_TEST_B='bravo value'",
                'export INFOMUX_TEST_C="charlie"',
            ]
        )
    )

    monkeypatch.delenv("INFOMUX_TEST_A", raising=False)
    monkeypatch.delenv("INFOMUX_TEST_B", raising=False)
    monkeypatch.delenv("INFOMUX_TEST_C", raising=False)

    loaded = load_dotenv(env_file)
    assert loaded == 3


def test_load_dotenv_does_not_override_existing(tmp_path: Path, monkeypatch) -> None:
    """Existing shell vars take precedence over .env entries."""
    env_file = tmp_path / ".env"
    env_file.write_text("INFOMUX_TEST_KEEP=from-file")

    monkeypatch.setenv("INFOMUX_TEST_KEEP", "from-shell")
    loaded = load_dotenv(env_file)

    assert loaded == 0


def test_load_dotenv_uses_cwd_default(tmp_path: Path, monkeypatch) -> None:
    """Defaults to loading .env in current working directory."""
    env_file = tmp_path / ".env"
    env_file.write_text("INFOMUX_TEST_CWD=yes")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INFOMUX_TEST_CWD", raising=False)

    loaded = load_dotenv()
    assert loaded == 1
