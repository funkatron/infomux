"""
Tests for the extract_text step.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infomux.steps.extract_text import (
    ExtractTextStep,
    HTMLTextExtractor,
    extract_text_from_html,
    is_html_file,
    run,
)
from infomux.steps import StepError


class TestHTMLTextExtractor:
    """Tests for HTML text extraction."""

    def test_extracts_text_from_simple_html(self) -> None:
        """Extracts text from simple HTML."""
        html = "<html><body><p>Hello world</p></body></html>"
        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        assert "Hello world" in text

    def test_skips_script_tags(self) -> None:
        """Skips content in script tags."""
        html = "<html><body><p>Hello</p><script>alert('hi')</script><p>World</p></body></html>"
        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        assert "Hello" in text
        assert "World" in text
        assert "alert" not in text

    def test_skips_style_tags(self) -> None:
        """Skips content in style tags."""
        html = "<html><body><p>Hello</p><style>body { color: red; }</style><p>World</p></body></html>"
        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        assert "Hello" in text
        assert "World" in text
        assert "color: red" not in text

    def test_preserves_structure(self) -> None:
        """Preserves some structure from HTML."""
        html = "<html><body><h1>Title</h1><p>Paragraph 1</p><p>Paragraph 2</p></body></html>"
        parser = HTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        assert "Title" in text
        assert "Paragraph 1" in text
        assert "Paragraph 2" in text


class TestExtractTextFromHTML:
    """Tests for extract_text_from_html function."""

    def test_simple_html(self) -> None:
        """Extracts text from simple HTML."""
        html = "<html><body><p>Hello world</p></body></html>"
        text = extract_text_from_html(html)
        assert "Hello world" in text
        assert "<" not in text  # No HTML tags

    def test_complex_html(self) -> None:
        """Extracts text from complex HTML."""
        html = """
        <html>
        <head><title>Test</title></head>
        <body>
            <h1>Main Title</h1>
            <p>This is a paragraph with <strong>bold</strong> text.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </body>
        </html>
        """
        text = extract_text_from_html(html)
        assert "Main Title" in text
        assert "This is a paragraph" in text
        assert "bold" in text
        assert "Item 1" in text
        assert "Item 2" in text


class TestIsHTMLFile:
    """Tests for HTML file detection."""

    def test_html_extension(self, tmp_path: Path) -> None:
        """Detects HTML by extension."""
        html_file = tmp_path / "test.html"
        html_file.write_text("<html></html>")
        assert is_html_file(html_file) is True

    def test_htm_extension(self, tmp_path: Path) -> None:
        """Detects HTM extension."""
        htm_file = tmp_path / "test.htm"
        htm_file.write_text("<html></html>")
        assert is_html_file(htm_file) is True

    def test_detects_by_content(self, tmp_path: Path) -> None:
        """Detects HTML by content even without extension."""
        html_file = tmp_path / "test.txt"
        html_file.write_text("<!DOCTYPE html><html><body>Hello</body></html>")
        assert is_html_file(html_file) is True

    def test_plain_text_not_html(self, tmp_path: Path) -> None:
        """Plain text files are not detected as HTML."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is plain text, not HTML.")
        assert is_html_file(text_file) is False


class TestExtractTextStep:
    """Tests for ExtractTextStep."""

    def test_step_name(self) -> None:
        """Step has correct name."""
        step = ExtractTextStep()
        assert step.name == "extract_text"

    def test_extracts_from_html_file(self, tmp_path: Path) -> None:
        """Extracts text from HTML file."""
        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><body><h1>Title</h1><p>Content here</p></body></html>"
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        step = ExtractTextStep()
        outputs = step.execute(html_file, output_dir)

        assert len(outputs) == 1
        transcript = outputs[0]
        assert transcript.name == "transcript.txt"
        content = transcript.read_text()
        assert "Title" in content
        assert "Content here" in content
        assert "<html>" not in content

    def test_handles_plain_text(self, tmp_path: Path) -> None:
        """Handles plain text files."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is plain text content.")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        step = ExtractTextStep()
        outputs = step.execute(text_file, output_dir)

        assert len(outputs) == 1
        transcript = outputs[0]
        content = transcript.read_text()
        assert "This is plain text content." in content

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """Raises error for missing file."""
        missing_file = tmp_path / "nonexistent.html"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        step = ExtractTextStep()
        with pytest.raises(StepError, match="not found"):
            step.execute(missing_file, output_dir)

    def test_raises_on_empty_content(self, tmp_path: Path) -> None:
        """Raises error for empty HTML."""
        html_file = tmp_path / "empty.html"
        html_file.write_text("<html><head></head><body></body></html>")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        step = ExtractTextStep()
        with pytest.raises(StepError, match="no text content"):
            step.execute(html_file, output_dir)


class TestRunFunction:
    """Tests for the run convenience function."""

    def test_returns_step_result(self, tmp_path: Path) -> None:
        """Returns StepResult on success."""
        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body><p>Hello</p></body></html>")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = run(html_file, output_dir)

        assert result.name == "extract_text"
        assert result.success is True
        assert len(result.outputs) == 1
        assert result.error is None

    def test_captures_error_on_failure(self, tmp_path: Path) -> None:
        """Captures error in StepResult on failure."""
        missing_file = tmp_path / "nonexistent.html"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = run(missing_file, output_dir)

        assert result.name == "extract_text"
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error
