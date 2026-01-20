"""
The 'run' command: process a media file through the pipeline.

This command takes a media file as input, creates a new job envelope,
and runs it through the configured pipeline steps. Each step produces
artifacts that are stored in the run directory.

Usage:
    infomux run input.mp4
    infomux run --pipeline transcribe input.mp4
    infomux run --steps extract_audio input.mp4
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
from pathlib import Path

from infomux.config import get_tool_paths
from infomux.job import InputFile, JobEnvelope, JobStatus, generate_run_id
from infomux.log import get_logger
from infomux.pipeline import run_pipeline
from infomux.pipeline_def import get_pipeline
from infomux.storage import get_run_dir, save_job

logger = get_logger(__name__)


def is_url(input_str: str) -> bool:
    """
    Check if the input string is a URL.

    Args:
        input_str: Input string to check.

    Returns:
        True if the input appears to be a URL, False otherwise.
    """
    parsed = urllib.parse.urlparse(input_str)
    return parsed.scheme in ("http", "https")


def download_url(url: str, output_path: Path) -> Path:
    """
    Download a file from a URL to a local path.

    Args:
        url: URL to download from.
        output_path: Local path to save the downloaded file.

    Returns:
        Path to the downloaded file.

    Raises:
        urllib.error.URLError: If the download fails.
        OSError: If the file cannot be written.
    """
    logger.info("downloading from URL: %s", url)
    try:
        with urllib.request.urlopen(url) as response:
            # Get content length if available
            content_length = response.headers.get("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                logger.info("file size: %.1f MB", size_mb)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download in chunks
            with open(output_path, "wb") as f:
                chunk_size = 8192
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if content_length:
                        percent = (downloaded / int(content_length)) * 100
                        logger.debug("downloaded: %.1f%%", percent)

        logger.info("downloaded to: %s", output_path)
        return output_path
    except urllib.error.HTTPError as e:
        raise urllib.error.URLError(f"HTTP error {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise urllib.error.URLError(f"Failed to download URL: {e.reason}") from e


def configure_parser(parser: ArgumentParser) -> None:
    """
    Configure the argument parser for the 'run' command.

    Args:
        parser: The subparser to configure.
    """
    parser.add_argument(
        "input",
        type=str,
        nargs="?",  # Optional when using --check-deps
        help="Path to the input media file or URL (http:// or https://). "
        "Supports audio (mp3, m4a, wav), video (mp4, mov), and text/html files. "
        "URLs are automatically downloaded to the run directory.",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        default=None,
        help="Pipeline to run (default: transcribe). "
        "Use 'infomux inspect --list-pipelines' to see available pipelines. "
        "HTML/text files automatically use 'web-summarize' pipeline unless overridden.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of steps to run (subset of pipeline). "
        "Only runs the specified steps and their dependencies. "
        "Example: --steps extract_audio,transcribe",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing. "
        "Displays the pipeline, steps, and job configuration that would be used.",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check for required dependencies (ffmpeg, whisper-cli, models) and exit. "
        "Shows status of all dependencies including optional ones like EasyOCR.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Ollama model for summarization steps. "
        "Overrides the default model. Example: qwen2.5:32b-instruct, llama3.2:3b",
    )
    parser.add_argument(
        "--content-type-hint",
        type=str,
        default=None,
        metavar="TYPE",
        help="Hint for content type to improve summarization quality. "
        "Options: meeting, talk, podcast, lecture, standup, 1on1, or any custom string. "
        "This affects the prompt used for LLM summarization.",
    )
    parser.add_argument(
        "--word-level-subtitles",
        action="store_true",
        help="Generate word-level subtitles where each word appears individually. "
        "Experimental feature that creates more granular subtitle timing. "
        "Requires transcribe_timed step.",
    )
    parser.add_argument(
        "--video-background-image",
        type=Path,
        default=None,
        help="Background image for video generation (requires audio-to-video pipeline). "
        "Path to an image file to use as background when generating video from audio.",
    )
    parser.add_argument(
        "--video-background-color",
        type=str,
        default=None,
        help="Background color for video generation (default: black). "
        "Can be a color name (e.g., 'blue') or hex code (e.g., '#FF0000'). "
        "Requires audio-to-video pipeline.",
    )
    parser.add_argument(
        "--video-size",
        type=str,
        default=None,
        metavar="WxH",
        help="Video dimensions for video generation (default: 1920x1080). "
        "Format: WIDTHxHEIGHT (e.g., 1280x720). Requires audio-to-video pipeline.",
    )
    parser.add_argument(
        "--lyric-font-name",
        type=str,
        default=None,
        help="Font family name for lyric video (default: Arial). "
        "Requires lyric-video pipeline.",
    )
    parser.add_argument(
        "--lyric-font-file",
        type=Path,
        default=None,
        help="Path to font file for lyric video (overrides --lyric-font-name). "
        "Requires lyric-video pipeline.",
    )
    parser.add_argument(
        "--lyric-font-size",
        type=int,
        default=None,
        help="Font size in pixels for lyric video (default: 48). "
        "Requires lyric-video pipeline.",
    )
    parser.add_argument(
        "--lyrics-file",
        type=Path,
        default=None,
        help="Path to official lyrics text file for forced alignment. "
        "When provided, uses align_lyrics step instead of transcribe_timed. "
        "If not provided, looks for lyrics.txt in the run directory. "
        "Requires lyric-video-aligned or lyric-video-aligned-isolated pipeline.",
    )
    parser.add_argument(
        "--lyric-font-color",
        type=str,
        default=None,
        help="Text color for lyric video (default: white). "
        "Can be a color name (e.g., 'yellow') or hex code (e.g., '#FFFF00'). "
        "Requires lyric-video pipeline.",
    )
    parser.add_argument(
        "--lyric-position",
        type=str,
        default=None,
        choices=["top", "center", "bottom"],
        help="Vertical position for lyrics (default: center). "
        "Requires lyric-video pipeline.",
    )
    parser.add_argument(
        "--lyric-word-spacing",
        type=int,
        default=None,
        help="Horizontal spacing between words in pixels (default: 20). "
        "Requires lyric-video pipeline.",
    )


def execute(args: Namespace) -> int:
    """
    Execute the 'run' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Check dependencies mode
    if args.check_deps:
        return _check_dependencies()

    # Require input file for normal run
    if not args.input:
        logger.error("input file is required")
        return 1

    input_str: str = args.input
    original_url: str | None = None
    input_path: Path
    run_id: str | None = None

    # Handle URL input
    if is_url(input_str):
        original_url = input_str
        logger.info("input is a URL: %s", original_url)

        # Generate run_id early so we can download to the run directory
        run_id = generate_run_id()
        run_dir = get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename from URL or use a default
        parsed_url = urllib.parse.urlparse(original_url)
        url_filename = Path(parsed_url.path).name
        if not url_filename or url_filename == "/":
            # No filename in URL, use a default based on content type
            url_filename = "input_media"

        # Download to run directory
        downloaded_path = run_dir / url_filename
        try:
            download_url(original_url, downloaded_path)
            input_path = downloaded_path
        except urllib.error.URLError as e:
            logger.error("failed to download URL: %s", e)
            return 1
        except Exception as e:
            logger.error("unexpected error downloading URL: %s", e)
            return 1
    else:
        # Handle local file path
        input_path = Path(input_str)

        # Validate input file
        if not input_path.exists():
            logger.error("input file not found: %s", input_path)
            return 1

        if not input_path.is_file():
            logger.error("input path is not a file: %s", input_path)
            return 1

    logger.info("processing input: %s", input_path)

    # Detect if input is HTML/text content and suggest web-summarize pipeline
    from infomux.steps.extract_text import is_html_file

    is_html = is_html_file(input_path)
    if is_html and args.pipeline is None:
        # Auto-detect HTML content and use web-summarize pipeline
        logger.info(
            "detected HTML/text content, using 'web-summarize' pipeline "
            "(use --pipeline to override)"
        )
        args.pipeline = "web-summarize"

    # Get pipeline definition
    try:
        pipeline = get_pipeline(args.pipeline)
    except ValueError as e:
        logger.error(str(e))
        return 1

    if args.pipeline is None:
        logger.info(
            "using default pipeline '%s' (%s)",
            pipeline.name,
            " → ".join(pipeline.step_names()),
        )
    else:
        logger.info("using pipeline: %s", pipeline.name)

    # Create input file metadata
    try:
        input_file = InputFile.from_path(input_path)
        if original_url:
            input_file.original_url = original_url
        logger.debug("input sha256: %s", input_file.sha256)
        logger.debug("input size: %d bytes", input_file.size_bytes)
        if original_url:
            logger.debug("original URL: %s", original_url)
    except Exception as e:
        logger.error("failed to read input file: %s", e)
        return 1

    # Create job envelope
    # If we already created a run_id for URL download, reuse it
    if run_id is not None:
        now = datetime.now(UTC).isoformat()
        job = JobEnvelope(
            id=run_id,
            created_at=now,
            updated_at=now,
            status=JobStatus.PENDING.value,
            input=input_file,
        )
    else:
        job = JobEnvelope.create(input_file=input_file)

    # Parse steps if specified (subset of pipeline)
    step_names = None
    if args.steps:
        step_names = [s.strip() for s in args.steps.split(",")]
        # Validate steps exist in pipeline
        valid_steps = set(pipeline.step_names())
        invalid = [s for s in step_names if s not in valid_steps]
        if invalid:
            logger.error(
                "unknown steps: %s (available: %s)",
                invalid,
                list(valid_steps),
            )
            return 1
        logger.info("running steps: %s", step_names)

    if args.dry_run:
        logger.info("dry run mode - not executing")
        # Show pipeline info
        print(f"Pipeline: {pipeline.name}")
        print(f"Description: {pipeline.description}")
        print(f"Steps: {pipeline.step_names()}")
        if step_names:
            print(f"Selected: {step_names}")
        print()
        print(job.to_json())
        return 0

    # Validate dependencies before starting
    tools = get_tool_paths()
    errors = tools.validate()
    if errors:
        for error in errors:
            logger.error(error)
        logger.error("run 'infomux run --check-deps' for more information")
        return 1

    # Save the job envelope
    job.update_status(JobStatus.RUNNING)
    save_job(job)
    # Get run_dir (may already exist if we downloaded a URL)
    run_dir = get_run_dir(job.id)
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("created run: %s", job.id)
    logger.debug("run directory: %s", run_dir)

    # Set model override if specified
    if args.model:
        import os
        os.environ["INFOMUX_OLLAMA_MODEL"] = args.model
        logger.debug("using model: %s", args.model)

    # Set content type hint if specified
    if args.content_type_hint:
        import os
        os.environ["INFOMUX_CONTENT_TYPE_HINT"] = args.content_type_hint
        logger.debug("content type hint: %s", args.content_type_hint)

    # Build step config overrides from CLI args
    step_configs = {}

    # Lyrics file for forced alignment
    if args.lyrics_file:
        if not args.lyrics_file.exists():
            logger.error("lyrics file not found: %s", args.lyrics_file)
            return 1
        step_configs["align_lyrics"] = {"lyrics_file": str(args.lyrics_file)}

    # Word-level subtitles config
    if args.word_level_subtitles:
        step_configs["transcribe_timed"] = {"generate_word_level": True}

    # Video generation config
    if args.video_background_image or args.video_background_color or args.video_size:
        generate_video_config = step_configs.get("generate_video", {})
        if args.video_background_image:
            generate_video_config["background_image"] = str(args.video_background_image)
        if args.video_background_color:
            generate_video_config["background_color"] = args.video_background_color
        if args.video_size:
            generate_video_config["video_size"] = args.video_size
        step_configs["generate_video"] = generate_video_config

    # Lyric video generation config
    if any([
        args.lyric_font_name,
        args.lyric_font_file,
        args.lyric_font_size,
        args.lyric_font_color,
        args.lyric_position,
        args.lyric_word_spacing,
    ]):
        lyric_video_config = step_configs.get("generate_lyric_video", {})
        if args.lyric_font_name:
            lyric_video_config["font_name"] = args.lyric_font_name
        if args.lyric_font_file:
            lyric_video_config["font_file"] = str(args.lyric_font_file)
        if args.lyric_font_size:
            lyric_video_config["font_size"] = args.lyric_font_size
        if args.lyric_font_color:
            lyric_video_config["font_color"] = args.lyric_font_color
        if args.lyric_position:
            lyric_video_config["position"] = args.lyric_position
        if args.lyric_word_spacing:
            lyric_video_config["word_spacing"] = args.lyric_word_spacing
        step_configs["generate_lyric_video"] = lyric_video_config

    # Also allow video-size and background-color to apply to lyric-video
    if args.video_size or args.video_background_color:
        lyric_video_config = step_configs.get("generate_lyric_video", {})
        if args.video_size:
            lyric_video_config["video_size"] = args.video_size
        if args.video_background_color:
            lyric_video_config["background_color"] = args.video_background_color
        step_configs["generate_lyric_video"] = lyric_video_config

    # Execute pipeline
    success = run_pipeline(
        job, run_dir, pipeline=pipeline, step_names=step_names, step_configs=step_configs
    )

    # Update final status
    if success:
        job.update_status(JobStatus.COMPLETED)
        logger.info("run completed: %s", job.id)
    else:
        # Status already set by pipeline on failure
        logger.error("run failed: %s", job.id)

    # Save final state
    save_job(job)

    # Output run directory path to stdout for scripting
    print(run_dir, file=sys.stdout)

    return 0 if success else 1


def _check_dependencies() -> int:
    """
    Check for required external dependencies.

    Returns:
        Exit code (0 if all deps found, 1 if any missing).
    """
    tools = get_tool_paths()

    print("Checking dependencies...")
    print()

    # ffmpeg
    if tools.ffmpeg:
        print(f"✓ ffmpeg: {tools.ffmpeg}")
    else:
        print("✗ ffmpeg: NOT FOUND")
        print("  Install: brew install ffmpeg")

    # whisper-cli
    if tools.whisper_cli:
        print(f"✓ whisper-cli: {tools.whisper_cli}")
    else:
        print("✗ whisper-cli: NOT FOUND")
        print("  Install: brew install whisper-cpp")

    # whisper model
    if tools.whisper_model:
        size_mb = tools.whisper_model.stat().st_size / (1024 * 1024)
        print(f"✓ whisper model: {tools.whisper_model} ({size_mb:.1f} MB)")
    else:
        print("✗ whisper model: NOT FOUND")
        print("  Download:")
        print("    mkdir -p ~/.local/share/infomux/models/whisper")
        print("    curl -L -o ~/.local/share/infomux/models/whisper/ggml-base.en.bin")
        print("      https://huggingface.co/.../ggml-base.en.bin")

    # OCR engines (for image text extraction)
    if tools.tesseract:
        print(f"✓ tesseract: {tools.tesseract} (default OCR engine)")
    else:
        print("✗ tesseract: NOT FOUND (required for OCR)")
        print("  Install: brew install tesseract")

    # EasyOCR (optional, better quality with GPU support)
    easyocr_available = False
    try:
        import easyocr
        import torch
        # Check for GPU acceleration
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
        gpu_status = ""
        if has_mps:
            gpu_status = " (Metal/Apple Silicon GPU)"
        elif has_cuda:
            gpu_status = " (CUDA GPU)"
        else:
            gpu_status = " (CPU only)"

        easyocr_available = True
        print(f"✓ EasyOCR: available{gpu_status} (optional, better quality)")
    except ImportError:
        print("○ EasyOCR: NOT FOUND (optional, for better OCR quality)")
        print("  Install: pip install easyocr")
        print("  For Apple Silicon GPU: pip install torch torchvision (Metal support)")


    print()

    errors = tools.validate()
    if errors:
        print(f"Missing {len(errors)} dependency(ies)")
        return 1
    else:
        print("All dependencies found!")
        return 0
