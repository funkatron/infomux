"""
The 'stream' command: real-time audio capture and pipeline processing.

Uses whisper-stream for live transcription from an audio input device,
then optionally processes the captured audio through a pipeline.

Usage:
    infomux stream                      # Capture and transcribe
    infomux stream --pipeline summarize # Capture, transcribe, summarize
    infomux stream --device 3           # Use device 3 directly
    infomux stream --list-devices       # List available devices
    infomux stream --duration 60        # Stop after 60 seconds
    infomux stream --silence 5          # Stop after 5 seconds of silence
    infomux stream --stop-word "stop"   # Stop when "stop" is detected
"""

from __future__ import annotations

import re
import signal
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from infomux.audio import (
    AudioDevice,
    get_device_by_id,
    list_audio_devices,
    select_audio_device,
)
from infomux.config import find_tool, get_whisper_model_path
from infomux.job import InputFile, JobEnvelope, JobStatus
from infomux.log import get_logger
from infomux.pipeline import run_pipeline
from infomux.pipeline_def import get_pipeline, list_pipelines
from infomux.storage import get_run_dir, save_job

logger = get_logger(__name__)


def configure_parser(parser: ArgumentParser) -> None:
    """
    Configure the argument parser for the 'stream' command.

    Args:
        parser: The subparser to configure.
    """
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        default=None,
        help="Audio device ID (skip interactive selection)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--no-save-audio",
        action="store_true",
        help="Don't save the recorded audio (disables precise timestamps)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en",
        help="Spoken language (default: en)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Stop recording after N seconds",
    )
    parser.add_argument(
        "--silence",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Stop recording after N seconds of silence",
    )
    parser.add_argument(
        "--stop-word",
        type=str,
        default="stop recording",
        metavar="PHRASE",
        help="Stop recording when this phrase is detected (default: 'stop recording')",
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        default=None,
        metavar="NAME",
        help="Pipeline to run after capture (e.g., transcribe, summarize)",
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List available pipelines and exit",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Ollama model for summarization (e.g., qwen2.5:32b-instruct)",
    )
    parser.add_argument(
        "--content-type-hint",
        type=str,
        default=None,
        metavar="TYPE",
        help="Hint for content type: meeting, talk, podcast, lecture, standup, 1on1, or custom",
    )


class StreamMonitor:
    """
    Monitors whisper-stream output for stop conditions.

    Tracks:
    - Duration elapsed
    - Time since last speech
    - Stop word detection
    """

    def __init__(
        self,
        process: subprocess.Popen,
        duration: int | None = None,
        silence: int | None = None,
        stop_word: str | None = None,
    ):
        self.process = process
        self.duration = duration
        self.silence = silence
        self.stop_word = stop_word.lower() if stop_word else None

        self.start_time = time.time()
        self.last_speech_time = time.time()
        self.should_stop = False
        self.stop_reason: str | None = None

        self._lock = threading.Lock()

    def check_duration(self) -> bool:
        """Check if duration limit exceeded."""
        if self.duration is None:
            return False
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration:
            with self._lock:
                self.should_stop = True
                self.stop_reason = f"duration limit ({self.duration}s)"
            return True
        return False

    def check_silence(self) -> bool:
        """Check if silence threshold exceeded."""
        if self.silence is None:
            return False
        silent_for = time.time() - self.last_speech_time
        if silent_for >= self.silence:
            with self._lock:
                self.should_stop = True
                self.stop_reason = f"silence ({self.silence}s)"
            return True
        return False

    def check_stop_word(self, text: str) -> bool:
        """Check if stop word detected in text."""
        if self.stop_word is None:
            return False
        if self.stop_word in text.lower():
            with self._lock:
                self.should_stop = True
                self.stop_reason = f"stop word '{self.stop_word}'"
            return True
        return False

    def on_speech(self, text: str) -> None:
        """Called when speech is detected."""
        self.last_speech_time = time.time()
        self.check_stop_word(text)

    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        return time.time() - self.start_time

    def remaining(self) -> float | None:
        """Return remaining time if duration set."""
        if self.duration is None:
            return None
        return max(0, self.duration - self.elapsed())


def _print_instructions(
    device: AudioDevice,
    duration: int | None,
    silence: int | None,
    stop_word: str | None,
) -> None:
    """Print clear instructions for the user."""
    print(file=sys.stderr)
    print("─" * 50, file=sys.stderr)
    print(f"  Recording from: {device.name}", file=sys.stderr)
    print(file=sys.stderr)
    print("  Stop recording by:", file=sys.stderr)
    print("    • Press Ctrl+C", file=sys.stderr)
    if duration:
        print(f"    • Wait {duration} seconds (auto-stop)", file=sys.stderr)
    if silence:
        print(f"    • Stay silent for {silence} seconds", file=sys.stderr)
    if stop_word:
        print(f"    • Say \"{stop_word}\"", file=sys.stderr)
    print("─" * 50, file=sys.stderr)
    print(file=sys.stderr)


def execute(args: Namespace) -> int:
    """
    Execute the 'stream' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # List modes
    if args.list_devices:
        return _list_devices()

    if args.list_pipelines:
        return _list_pipelines_cmd()

    # Validate pipeline if specified
    pipeline = None
    if args.pipeline:
        try:
            pipeline = get_pipeline(args.pipeline)
            logger.info("will run pipeline '%s' after capture", pipeline.name)
        except ValueError as e:
            logger.error(str(e))
            return 1

    # Find whisper-stream
    whisper_stream = find_tool("whisper-stream")
    if not whisper_stream:
        logger.error("whisper-stream not found")
        logger.error("Install via: brew install whisper-cpp")
        return 1

    # Get whisper model
    model_path = get_whisper_model_path()
    if not model_path:
        logger.error("Whisper model not found")
        logger.error("Set INFOMUX_WHISPER_MODEL or download a model")
        return 1

    # Get device
    device: AudioDevice | None = None

    if args.device is not None:
        # Direct device selection
        device = get_device_by_id(args.device)
        if not device:
            logger.error("Device %d not found", args.device)
            logger.info("Use --list-devices to see available devices")
            return 1
    else:
        # Interactive selection
        try:
            devices = list_audio_devices()
        except RuntimeError as e:
            logger.error("Failed to list devices: %s", e)
            return 1

        device = select_audio_device(devices)
        if not device:
            return 1

    # Show clear instructions
    _print_instructions(device, args.duration, args.silence, args.stop_word)

    # Create job envelope for this stream
    job = JobEnvelope.create()
    job.config["stream"] = True
    job.config["device_id"] = device.id
    job.config["device_name"] = device.name
    job.config["language"] = args.language
    if args.duration:
        job.config["duration"] = args.duration
    if args.silence:
        job.config["silence"] = args.silence
    if args.stop_word:
        job.config["stop_word"] = args.stop_word

    run_dir = get_run_dir(job.id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build whisper-stream command
    transcript_path = run_dir / "transcript.txt"
    has_stop_conditions = args.duration or args.silence or args.stop_word

    cmd = [
        str(whisper_stream),
        "-m", str(model_path),
        "-c", str(device.id),
        "-l", args.language,
    ]

    # Only use -f if no monitoring (monitoring writes its own timestamped file)
    if not has_stop_conditions:
        cmd.extend(["-f", str(transcript_path)])

    # Always save audio by default (needed for precise timestamps)
    save_audio = not getattr(args, "no_save_audio", False)
    if save_audio:
        cmd.append("--save-audio")
        job.config["save_audio"] = True

    logger.debug("running: %s", " ".join(cmd))

    # Update job status
    job.update_status(JobStatus.RUNNING)
    save_job(job)

    # Run whisper-stream with monitoring
    return_code = _run_with_monitoring(
        cmd=cmd,
        transcript_path=transcript_path if has_stop_conditions else None,
        duration=args.duration,
        silence=args.silence,
        stop_word=args.stop_word,
    )

    # Update job status
    if return_code != 0:
        job.update_status(JobStatus.FAILED, f"whisper-stream exited {return_code}")
        save_job(job)
        return return_code

    # Post-process captured audio
    if save_audio:
        audio_file = _find_saved_audio(run_dir)
        if audio_file:
            job.artifacts.append(str(audio_file))

            if pipeline:
                # Run the specified pipeline on the captured audio
                logger.info("Running pipeline '%s' on captured audio...", pipeline.name)

                # Update job with input info for pipeline
                job.input = InputFile.from_path(audio_file)

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

                # Run pipeline (skip steps that don't apply to audio-only)
                skip_steps = {"extract_audio", "embed_subs"}  # No video input
                steps_to_run = [
                    s.name for s in pipeline.steps if s.name not in skip_steps
                ]
                success = run_pipeline(
                    job=job,
                    run_dir=run_dir,
                    pipeline=pipeline,
                    step_names=steps_to_run,
                )
                if not success:
                    save_job(job)
                    return 1
            else:
                # Default: just get precise timestamps
                logger.info("Post-processing for precise timestamps...")
                timestamp_result = _run_whisper_cli_timestamps(
                    audio_file=audio_file,
                    model_path=model_path,
                    language=args.language,
                    run_dir=run_dir,
                )
                if timestamp_result:
                    for artifact in timestamp_result:
                        job.artifacts.append(str(artifact))
                        logger.info("Created: %s", artifact.name)

    # Add rough transcript if it exists
    if transcript_path.exists():
        job.artifacts.append(str(transcript_path))

    job.update_status(JobStatus.COMPLETED)
    save_job(job)

    # Output run directory
    print(run_dir, file=sys.stdout)

    return 0


def _run_with_monitoring(
    cmd: list[str],
    transcript_path: Path | None,
    duration: int | None,
    silence: int | None,
    stop_word: str | None,
) -> int:
    """
    Run whisper-stream with optional stop conditions.

    Args:
        cmd: Command to run.
        transcript_path: Path to write transcript (None = whisper-stream writes it).
        duration: Max duration in seconds (None = unlimited).
        silence: Stop after N seconds of silence (None = disabled).
        stop_word: Stop when this phrase is detected (None = disabled).

    Returns:
        Exit code.
    """
    # If no special stop conditions, run simply
    if duration is None and silence is None and stop_word is None:
        return _run_simple(cmd)

    # Run with monitoring
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    monitor = StreamMonitor(
        process=process,
        duration=duration,
        silence=silence,
        stop_word=stop_word,
    )

    try:
        while process.poll() is None:
            # Check duration
            if monitor.check_duration():
                logger.info("Stopping: %s", monitor.stop_reason)
                break

            # Check silence
            if monitor.check_silence():
                logger.info("Stopping: %s", monitor.stop_reason)
                break

            # Read output (non-blocking would be better but this works)
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    # Echo to user
                    print(line, end="", flush=True)

                    # Extract speech content
                    text = line.strip()
                    # Skip empty, control sequences, and blank audio markers
                    is_blank = "[BLANK_AUDIO]" in text or text.startswith("[2K")
                    if text and not is_blank:
                        # Remove timestamp prefix like [00:00:00.000 --> ...]
                        text = re.sub(r"^\[[\d:.,\s\->]+\]\s*", "", text)
                        if text:
                            monitor.on_speech(text)

                            # Check if stop word triggered
                            if monitor.should_stop:
                                logger.info("Stopping: %s", monitor.stop_reason)
                                break

            # Small sleep to prevent busy-waiting
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Stopping: user interrupt")

    # Gracefully stop the process
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    return 0


def _run_simple(cmd: list[str]) -> int:
    """Run whisper-stream without monitoring (Ctrl+C only)."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        process.wait()
        return process.returncode

    except KeyboardInterrupt:
        logger.info("Stopping: user interrupt")
        process.send_signal(signal.SIGINT)
        process.wait()
        return 0


def _list_devices() -> int:
    """
    List available audio devices.

    Returns:
        Exit code (0 for success).
    """
    try:
        devices = list_audio_devices()
    except RuntimeError as e:
        logger.error("Failed to list devices: %s", e)
        return 1

    if not devices:
        print("No audio devices found.")
        return 0

    print("Available audio devices:")
    print()
    for device in devices:
        print(f"  [{device.id}] {device.name}")
    print()
    print("Use: infomux stream --device <id>")

    return 0


def _list_pipelines_cmd() -> int:
    """
    List available pipelines.

    Returns:
        Exit code (0 for success).
    """
    print("Available pipelines for stream:")
    print()
    skip_steps = {"extract_audio", "embed_subs"}  # Don't apply to audio-only
    for name in list_pipelines():
        pipeline = get_pipeline(name)
        # Show pipelines that make sense for audio input
        steps = [s.name for s in pipeline.steps if s.name not in skip_steps]
        if steps:
            print(f"  {name}")
            print(f"    {pipeline.description}")
            print(f"    Steps: {' → '.join(steps)}")
            print()

    print("Use: infomux stream --pipeline <name>")
    return 0


def _find_saved_audio(run_dir: Path) -> Path | None:
    """
    Find the audio file saved by whisper-stream.

    whisper-stream saves audio with timestamp names like YYYYMMDDHHMMSS.wav
    in the current directory.
    """
    import glob

    # Look for timestamp-named wav files in cwd (most recent)
    wav_files = sorted(glob.glob("*.wav"), reverse=True)
    if wav_files:
        src = Path(wav_files[0])
        dest = run_dir / "audio.wav"
        src.rename(dest)
        return dest

    # Also check run_dir
    for pattern in ["audio.wav", "output.wav", "*.wav"]:
        matches = list(run_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def _run_whisper_cli_timestamps(
    audio_file: Path,
    model_path: Path,
    language: str,
    run_dir: Path,
) -> list[Path] | None:
    """
    Run whisper-cli on audio file to get precise timestamps.

    Generates:
    - transcript.json: Full JSON with segment timestamps
    - transcript.srt: SRT subtitles
    - transcript.vtt: VTT subtitles

    Returns:
        List of created artifact paths, or None on failure.
    """
    whisper_cli = find_tool("whisper-cli")
    if not whisper_cli:
        logger.warning("whisper-cli not found, skipping precise timestamps")
        return None

    output_prefix = run_dir / "transcript"

    cmd = [
        str(whisper_cli),
        "-m", str(model_path),
        "-l", language,
        "-f", str(audio_file),
        "-of", str(output_prefix),
        "-ojf",  # Full JSON with timestamps
        "-osrt",  # SRT subtitles
        "-ovtt",  # VTT subtitles
    ]

    logger.debug("running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning("whisper-cli failed: %s", result.stderr)
            return None

    except Exception as e:
        logger.warning("whisper-cli error: %s", e)
        return None

    # Collect created files
    artifacts = []
    for ext in [".json", ".srt", ".vtt"]:
        path = Path(str(output_prefix) + ext)
        if path.exists():
            artifacts.append(path)

    return artifacts if artifacts else None
