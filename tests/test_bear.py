from unittest.mock import MagicMock, patch

from infomux.bear import create_note, get_tags_from_env, parse_tags, replace_note


def test_parse_tags() -> None:
    assert parse_tags("a, b ,c") == ["a", "b", "c"]


def test_get_tags_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TEST_BEAR_TAGS", "one,two")
    assert get_tags_from_env("TEST_BEAR_TAGS", "default") == ["one", "two"]


@patch("infomux.bear.subprocess.run")
def test_create_note_builds_bear_url(mock_run) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    create_note("Title", "Body", ["tag"], open_note=True)
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "open"
    assert "bear://x-callback-url/create" in call_args[1]
    assert "open_note=yes" in call_args[1]


@patch("infomux.bear.subprocess.run")
def test_replace_note_uses_add_text_replace(mock_run) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    replace_note("Title", "Body", ["tag"], open_note=False)
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "open"
    assert "bear://x-callback-url/add-text" in call_args[1]
    assert "mode=replace" in call_args[1]
