"""
Extract text step: extract plain text from HTML, text files, or images.

This step handles multiple input formats:
- HTML files (from web pages): Extracts text, skipping scripts/styles
- Plain text files: Uses content directly
- Image files: OCR extraction
  - Default: Tesseract (CLI-based, lightweight, required)
  - Optional: EasyOCR (better quality, GPU-accelerated on Apple Silicon)

All formats output transcript.txt for use in downstream steps like summarize.
OCR output is automatically cleaned to handle common artifacts.
"""

from __future__ import annotations

import html.parser
import re
import time
from dataclasses import dataclass
from pathlib import Path

from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step

logger = get_logger(__name__)

# Output filename for extracted text
TRANSCRIPT_FILENAME = "transcript.txt"


class HTMLTextExtractor(html.parser.HTMLParser):
    """HTML parser that extracts text content, skipping script/style tags."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.skip_tags = {"script", "style", "noscript", "meta", "head"}
        self.current_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track when we enter a tag to skip."""
        self.current_tag = tag.lower()

    def handle_endtag(self, tag: str) -> None:
        """Track when we exit a tag."""
        if tag.lower() == self.current_tag:
            self.current_tag = None
        # Add newline after block elements
        if tag.lower() in {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Extract text data, skipping script/style content."""
        if self.current_tag not in self.skip_tags:
            # Clean up whitespace but preserve structure
            cleaned = data.strip()
            if cleaned:
                self.text_parts.append(cleaned)

    def get_text(self) -> str:
        """Get the extracted text with normalized whitespace."""
        text = " ".join(self.text_parts)
        # Normalize whitespace: collapse multiple spaces/newlines
        text = re.sub(r"\s+", " ", text)
        # Restore paragraph breaks
        text = re.sub(r" \n ", "\n\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def extract_text_from_html(html_content: str) -> str:
    """
    Extract plain text from HTML content.

    Args:
        html_content: HTML content as string.

    Returns:
        Extracted plain text.
    """
    parser = HTMLTextExtractor()
    parser.feed(html_content)
    return parser.get_text()


def is_html_file(file_path: Path) -> bool:
    """
    Check if a file appears to be HTML.

    Args:
        file_path: Path to the file.

    Returns:
        True if the file appears to be HTML.
    """
    # Check extension
    ext = file_path.suffix.lower()
    if ext in {".html", ".htm"}:
        return True

    # Check content-type by reading first bytes
    try:
        with open(file_path, "rb") as f:
            first_bytes = f.read(512)
            # Look for HTML tags
            first_chars = first_bytes.decode("utf-8", errors="ignore").lower()
            if "<html" in first_chars or "<!doctype html" in first_chars:
                return True
    except Exception:
        pass

    return False


def is_image_file(file_path: Path) -> bool:
    """
    Check if a file is an image that could be processed with OCR.

    Args:
        file_path: Path to the file.

    Returns:
        True if the file appears to be an image.
    """
    # Check extension
    ext = file_path.suffix.lower()
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
    return ext in image_extensions


def cleanup_ocr_text(text: str, aggressive: bool = False) -> str:
    """
    Clean up text extracted from OCR.

    OCR output can be messy with:
    - Extra whitespace
    - Line breaks in the middle of words
    - Inconsistent spacing
    - Special characters that should be normalized
    - Noise lines (when aggressive=True)

    Args:
        text: Raw OCR text.
        aggressive: If True, apply aggressive line filtering (removes noise lines).
                    Useful for screenshots or images with UI elements.

    Returns:
        Cleaned text.
    """
    # Basic cleanup: normalize whitespace and structure
    text = re.sub(r"\s+", " ", text)
    # Remove line breaks that split words (heuristic: single char before newline)
    text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)
    # Normalize multiple newlines to double newline (paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Aggressive filtering: remove noise lines (inspired by questlog's filter_ocr_cruft)
    if aggressive:
        lines = text.split("\n")
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) < 3:
                continue

            # Count character types
            letters = len(re.findall(r"[a-zA-Z]", line_stripped))
            digits = len(re.findall(r"\d", line_stripped))
            symbols = len(re.findall(r"[^\w\s]", line_stripped))
            total_chars = len(line_stripped)

            # Skip lines that are mostly symbols (>35% symbols)
            if total_chars > 0 and symbols / total_chars > 0.35:
                continue

            # Skip lines with too few letters (<40% letters for lines >5 chars)
            if total_chars > 5 and letters / total_chars < 0.4:
                continue

            # Skip lines that are mostly numbers (>50% digits for lines >8 chars)
            if total_chars > 8 and digits / total_chars > 0.5:
                continue

            # Keep the line
            filtered_lines.append(line_stripped)

        text = "\n".join(filtered_lines)

    # Final trim
    text = text.strip()
    return text


def _try_easyocr(image_path: Path) -> str | None:
    """
    Try to extract text using EasyOCR (deep learning OCR, better quality).

    Optional dependency - only used if installed. Supports GPU acceleration
    on Apple Silicon when PyTorch with Metal support is available.

    Args:
        image_path: Path to the image file.

    Returns:
        Extracted text if EasyOCR is available and succeeds, None otherwise.
    """
    try:
        import easyocr
    except ImportError:
        logger.debug("EasyOCR not available (install with: pip install easyocr)")
        return None

    try:
        # Detect if GPU/Metal acceleration is available
        # On Apple Silicon, PyTorch with Metal support will use GPU automatically
        use_gpu = False
        try:
            import torch
            # Check for Metal (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                use_gpu = True
                logger.debug("EasyOCR using GPU acceleration (Metal/Apple Silicon)")
            # Check for CUDA (Linux/Windows)
            elif torch.cuda.is_available():
                use_gpu = True
                logger.debug("EasyOCR using GPU acceleration (CUDA)")
            else:
                logger.debug("EasyOCR using CPU (no GPU available)")
        except ImportError:
            # torch not available, use CPU
            logger.debug("EasyOCR using CPU (PyTorch not available)")
        
        # Initialize reader (cached globally for performance)
        # Note: First call loads model (~100MB), subsequent calls reuse it
        # gpu=True enables Metal on Apple Silicon or CUDA if available
        reader = easyocr.Reader(['en'], gpu=use_gpu)
        results = reader.readtext(str(image_path))

        # Extract text from results (filter by confidence)
        # Format: [(bbox, text, confidence), ...]
        lines = []
        for (bbox, text, confidence) in results:
            if confidence >= 0.5 and text.strip() and len(text.strip()) > 2:
                lines.append(text.strip())

        if lines:
            text = "\n".join(lines)
            logger.debug("EasyOCR extracted %d lines", len(lines))
            return text
        return None
    except Exception as e:
        logger.debug("EasyOCR failed: %s", e)
        return None


def _try_tesseract(image_path: Path) -> str | None:
    """
    Try to extract text using Tesseract OCR (fallback).

    Uses multiple PSM modes and combines results for better accuracy.

    Args:
        image_path: Path to the image file.

    Returns:
        Extracted text if Tesseract is available and succeeds, None otherwise.
    """
    from infomux.config import get_tool_paths
    import subprocess

    tools = get_tool_paths()
    if not tools.tesseract:
        logger.debug("Tesseract not available (install with: brew install tesseract)")
        return None

    try:
        # Try multiple PSM modes for better accuracy (from questlog)
        # PSM 6 = Uniform block of text (good for documents)
        # PSM 11 = Sparse text (good for scattered elements)
        # PSM 13 = Raw line (no layout analysis)
        psm_modes = [6, 11, 13]
        all_lines = set()

        for psm in psm_modes:
            try:
                cmd = [
                    str(tools.tesseract),
                    str(image_path),
                    "stdout",
                    "--psm", str(psm),
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout:
                    lines = [
                        line.strip()
                        for line in result.stdout.splitlines()
                        if line.strip()
                    ]
                    all_lines.update(lines)
            except Exception:
                continue

        if all_lines:
            text = "\n".join(sorted(all_lines))  # Sort for consistency
            logger.debug("Tesseract extracted %d unique lines using %d PSM modes", len(all_lines), len(psm_modes))
            return text
        return None
    except Exception as e:
        logger.debug("Tesseract failed: %s", e)
        return None


def extract_text_from_image(image_path: Path) -> str:
    """
    Extract text from an image using OCR.

    Uses Tesseract by default (CLI-based, lightweight). If EasyOCR is installed
    (optional dependency), uses it instead for better quality with GPU acceleration
    on Apple Silicon.

    Args:
        image_path: Path to the image file.

    Returns:
        Extracted text.

    Raises:
        StepError: If Tesseract is not available or both OCR engines fail.
    """
    # Try EasyOCR first if available (better quality, optional dependency)
    text = _try_easyocr(image_path)
    if text:
        return text

    # Use Tesseract (default, CLI-based)
    text = _try_tesseract(image_path)
    if text:
        return text

    # Neither worked
    raise StepError(
        "extract_text",
        "OCR failed: Tesseract not available or failed. "
        "Install: brew install tesseract. "
        "For better quality (optional): pip install easyocr",
    )


@register_step
@dataclass
class ExtractTextStep:
    """
    Pipeline step to extract text from HTML or text files.

    Takes HTML files (from web pages) or plain text files and extracts
    clean text content, outputting it as transcript.txt for use in
    downstream steps like summarize.
    """

    name: str = "extract_text"

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Extract text from HTML or text file.

        Args:
            input_path: Path to the input HTML or text file.
            output_dir: Directory to write the extracted text.

        Returns:
            List containing the path to the transcript file.

        Raises:
            StepError: If the file cannot be read or processed.
        """
        if not input_path.exists():
            raise StepError(self.name, f"input file not found: {input_path}")

        output_path = output_dir / TRANSCRIPT_FILENAME

        logger.info("extracting text from: %s", input_path.name)

        try:
            # Determine file type and extract text accordingly
            if is_image_file(input_path):
                # Extract text from image using OCR (Tesseract default, EasyOCR if available)
                logger.info("detected image file, extracting text with OCR")
                text = extract_text_from_image(input_path)
                # Clean up OCR output (can be messy)
                text = cleanup_ocr_text(text)
            elif is_html_file(input_path):
                # Read and extract text from HTML
                content = input_path.read_text(encoding="utf-8", errors="replace")
                logger.debug("detected HTML content, extracting text")
                text = extract_text_from_html(content)
            else:
                # Read as plain text
                content = input_path.read_text(encoding="utf-8", errors="replace")
                # Check if it's actually HTML by content
                if "<html" in content.lower()[:1000]:
                    logger.debug("detected HTML content in text file, extracting text")
                    text = extract_text_from_html(content)
                else:
                    logger.debug("treating as plain text")
                    text = content

            # Final cleanup
            text = text.strip()
            if not text:
                raise StepError(self.name, "no text content extracted from file")

            # Write output
            output_path.write_text(text, encoding="utf-8")

            text_size = len(text)
            logger.info(
                "extracted text: %s (%d chars)", output_path.name, text_size
            )

            # Log preview
            preview_lines = text.split("\n")[:3]
            preview = "\n".join(preview_lines)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            logger.debug("text preview:\n%s", preview)

            return [output_path]

        except UnicodeDecodeError as e:
            raise StepError(
                self.name, f"failed to decode file as UTF-8: {e}"
            )
        except Exception as e:
            raise StepError(self.name, f"failed to extract text: {e}")


def run(input_path: Path, output_dir: Path) -> StepResult:
    """
    Convenience function to run the extract_text step.

    Args:
        input_path: Path to input HTML or text file.
        output_dir: Directory for output artifacts.

    Returns:
        StepResult with execution details.
    """
    step = ExtractTextStep()
    start_time = time.monotonic()

    try:
        outputs = step.execute(input_path, output_dir)
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=True,
            outputs=outputs,
            duration_seconds=duration,
        )
    except StepError as e:
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=False,
            outputs=[],
            duration_seconds=duration,
            error=str(e),
        )
