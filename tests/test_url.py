"""
Tests for URL input support in the run command.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.commands.run import download_url, is_url
from infomux.job import InputFile


class TestIsUrl:
    """Tests for URL detection."""

    def test_http_url(self) -> None:
        """HTTP URLs are detected."""
        assert is_url("http://example.com/video.mp4") is True

    def test_https_url(self) -> None:
        """HTTPS URLs are detected."""
        assert is_url("https://example.com/video.mp4") is True

    def test_file_path(self) -> None:
        """File paths are not detected as URLs."""
        assert is_url("/path/to/file.mp4") is False
        assert is_url("file.mp4") is False
        assert is_url("./relative/path.mp4") is False

    def test_other_schemes(self) -> None:
        """Other URL schemes are not detected."""
        assert is_url("ftp://example.com/file.mp4") is False
        assert is_url("file:///path/to/file.mp4") is False


class TestDownloadUrl:
    """Tests for URL downloading."""

    @patch("infomux.commands.run.urllib.request.urlopen")
    def test_download_success(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        """URL downloads successfully."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "11"}
        mock_response.read.side_effect = [b"hello world", b""]
        mock_urlopen.return_value.__enter__.return_value = mock_response

        output_path = tmp_path / "downloaded.mp4"
        result = download_url("https://example.com/video.mp4", output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.read_bytes() == b"hello world"
        mock_urlopen.assert_called_once()

    @patch("infomux.commands.run.urllib.request.urlopen")
    def test_download_chunks(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        """URL downloads in chunks."""
        # Mock HTTP response with multiple chunks
        mock_response = MagicMock()
        mock_response.headers = {"Content-Length": "22"}
        mock_response.read.side_effect = [b"chunk1", b"chunk2", b""]
        mock_urlopen.return_value.__enter__.return_value = mock_response

        output_path = tmp_path / "downloaded.mp4"
        download_url("https://example.com/video.mp4", output_path)

        assert output_path.read_bytes() == b"chunk1chunk2"

    @patch("infomux.commands.run.urllib.request.urlopen")
    def test_download_http_error(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        """HTTP errors are raised."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com/video.mp4", 404, "Not Found", None, None
        )

        output_path = tmp_path / "downloaded.mp4"
        with pytest.raises(urllib.error.URLError, match="HTTP error"):
            download_url("https://example.com/video.mp4", output_path)

    @patch("infomux.commands.run.urllib.request.urlopen")
    def test_download_creates_directory(
        self, mock_urlopen: MagicMock, tmp_path: Path
    ) -> None:
        """Download creates parent directory if needed."""
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.read.return_value = b""
        mock_urlopen.return_value.__enter__.return_value = mock_response

        output_path = tmp_path / "subdir" / "downloaded.mp4"
        download_url("https://example.com/video.mp4", output_path)

        assert output_path.parent.exists()


class TestInputFileWithUrl:
    """Tests for InputFile with original_url."""

    def test_input_file_with_url(self, tmp_path: Path) -> None:
        """InputFile can store original_url."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"test content")

        input_file = InputFile.from_path(test_file)
        input_file.original_url = "https://example.com/video.mp4"

        assert input_file.original_url == "https://example.com/video.mp4"
        assert input_file.path == str(test_file)

    def test_input_file_json_roundtrip_with_url(
        self, tmp_path: Path
    ) -> None:
        """InputFile with URL round-trips through JSON."""
        from infomux.job import JobEnvelope

        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"test content")

        input_file = InputFile.from_path(test_file)
        input_file.original_url = "https://example.com/video.mp4"

        job = JobEnvelope.create(input_file=input_file)
        json_str = job.to_json()
        restored = JobEnvelope.from_json(json_str)

        assert restored.input is not None
        assert restored.input.original_url == "https://example.com/video.mp4"
        assert restored.input.path == str(test_file)
