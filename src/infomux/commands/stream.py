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
    get_audio_levels,
    get_default_input,
    get_default_loopback,
    get_default_output,
    get_device_by_id,
    list_audio_devices,
    list_input_devices,
    list_loopback_devices,
    list_output_devices,
    record_audio,
    render_level_meter,
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
        help="Audio device ID to use (skip interactive selection). "
        "Use 'infomux stream --list-devices' to see available device IDs. "
        "This selects an input device (backward compatible).",
    )
    parser.add_argument(
        "--input",
        type=int,
        default=None,
        help="Input device ID (-1 to disable, omit for default). "
        "By default, records from default input + default loopback. "
        "Use -1 to disable input recording.",
    )
    parser.add_argument(
        "--output",
        type=int,
        default=None,
        help="Output/loopback device ID (-1 to disable, omit for default). "
        "By default, records from default input + default loopback. "
        "Use -1 to disable system audio recording.",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Interactive device selection with live audio meters. "
        "Shows real-time audio levels for each device.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all available audio input devices with their IDs and exit. "
        "Use the ID with --device to select a specific microphone.",
    )
    parser.add_argument(
        "--no-save-audio",
        action="store_true",
        help="Don't save the recorded audio file. "
        "This disables precise word-level timestamps but saves disk space. "
        "Only the transcript files are saved.",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en",
        help="Spoken language code for transcription (default: en). "
        "Examples: en, es, fr, de, ja. Use the ISO 639-1 language code.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Stop recording after N seconds. "
        "Useful for time-limited recordings. Example: --duration 300 (5 minutes).",
    )
    parser.add_argument(
        "--silence",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Stop recording after N seconds of silence. "
        "Great for dictation - automatically stops when you finish speaking. "
        "Example: --silence 5 (stops after 5 seconds of silence).",
    )
    parser.add_argument(
        "--stop-word",
        type=str,
        default="stop recording",
        metavar="PHRASE",
        help=(
            "Stop recording when this phrase is detected in the transcript "
            "(default: 'stop recording'). "
            "The phrase must be spoken clearly and detected by the transcription. "
            "Example: --stop-word 'end note'"
        ),
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        default=None,
        metavar="NAME",
        help="Pipeline to run after recording completes. "
        "By default, only transcription is performed. "
        "Use 'summarize' to also generate an LLM summary. "
        "Use 'infomux inspect --list-pipelines' to see all options.",
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List available pipelines that can be used with --pipeline and exit.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Ollama model for summarization steps (if using --pipeline summarize). "
        "Overrides the default model. Example: qwen2.5:32b-instruct",
    )
    parser.add_argument(
        "--content-type-hint",
        type=str,
        default=None,
        metavar="TYPE",
        help=(
            "Hint for content type to improve summarization quality. "
            "Options: meeting, talk, podcast, lecture, standup, 1on1, "
            "or any custom string. "
            "Only used if --pipeline includes summarization."
        ),
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


def _print_instructions_multi(
    device_names: list[str],
    duration: int | None,
    silence: int | None,
    stop_word: str | None,
) -> None:
    """Print clear instructions for multiple devices."""
    print(file=sys.stderr)
    print("─" * 50, file=sys.stderr)
    print(f"  Recording from: {' + '.join(device_names)}", file=sys.stderr)
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

    # Determine which devices to use
    input_devices: list[AudioDevice] = []
    loopback_devices: list[AudioDevice] = []

    if args.prompt:
        # Interactive mode with live meters
        try:
            selected_inputs, selected_loopbacks = interactive_device_selection()
            input_devices = selected_inputs
            loopback_devices = selected_loopbacks
        except (KeyboardInterrupt, EOFError):
            return 1
    else:
        # Non-interactive mode
        # Handle backward compatibility: --device flag selects input only
        if args.device is not None:
            device = get_device_by_id(args.device)
            if not device:
                logger.error("Device %d not found", args.device)
                logger.info("Use --list-devices to see available devices")
                return 1
            input_devices = [device]
            # Backward compatible: don't record loopback when --device is used
            loopback_devices = []
        else:
            # New default behavior: record input + output
            # Logic: if flag is -1, disable that side; if None, use default; if set, use specified
            
            # Handle input device selection
            if args.input == -1:
                # Explicitly disabled
                logger.debug("Input recording disabled (--input -1)")
                input_devices = []
            elif args.input is not None:
                # Explicitly specified device ID
                device = get_device_by_id(args.input)
                if not device:
                    logger.error("Input device %d not found", args.input)
                    logger.info("Use --list-devices to see available devices")
                    return 1
                input_devices = [device]
                logger.debug("Using specified input device: %s", device.name)
            else:
                # Use default input (when args.input is None)
                default_input = get_default_input()
                if default_input:
                    input_devices = [default_input]
                    logger.debug("Using default input device: %s", default_input.name)
                else:
                    logger.warning("No default input device found")

            # Handle loopback device selection
            if args.output == -1:
                # Explicitly disabled
                logger.debug("Output recording disabled (--output -1)")
                loopback_devices = []
            elif args.output is not None:
                # Explicitly specified device ID
                device = get_device_by_id(args.output)
                if not device:
                    logger.error("Output device %d not found", args.output)
                    logger.info("Use --list-devices to see available devices")
                    return 1
                loopback_devices = [device]
                logger.debug("Using specified output device: %s", device.name)
            else:
                # Use default output device (preferring loopback)
                # First try loopback devices (for system audio capture)
                default_loopback = get_default_loopback()
                if default_loopback:
                    loopback_devices = [default_loopback]
                    logger.debug("Using default loopback device: %s", default_loopback.name)
                    logger.info(
                        "Note: To capture system audio, set '%s' as your system output device "
                        "in System Settings > Sound > Output",
                        default_loopback.name,
                    )
                else:
                    # Fall back to any output-capable device
                    default_output = get_default_output()
                    if default_output:
                        loopback_devices = [default_output]
                        logger.debug("Using default output device: %s", default_output.name)
                        logger.warning(
                            "No loopback device found. Using '%s' for output capture. "
                            "For system audio, install BlackHole or set a loopback device as system output.",
                            default_output.name,
                        )
                    else:
                        logger.warning("No output device found")
                        logger.info("Install BlackHole or similar loopback device to capture system audio")

    # Validate we have at least one device
    all_devices = input_devices + loopback_devices
    
    # Log what we selected for debugging
    logger.info("Selected devices for recording:")
    logger.info("  Input devices (%d): %s", len(input_devices), [f"{d.name} (id={d.id})" for d in input_devices])
    logger.info("  Loopback devices (%d): %s", len(loopback_devices), [f"{d.name} (id={d.id})" for d in loopback_devices])
    logger.info("  Total devices: %d", len(all_devices))
    
    if len(loopback_devices) == 0:
        logger.warning("No loopback devices selected - system audio will not be captured")
        logger.info("To capture system audio, ensure BlackHole or similar is installed and detected")
    
    if not all_devices:
        logger.error("No devices selected for recording")
        logger.info(
            "Use --prompt for interactive selection, or specify --input/--output"
        )
        return 1

    # Show clear instructions
    device_names = [d.name for d in all_devices]
    logger.info("Recording from: %s", " + ".join(device_names))
    _print_instructions_multi(device_names, args.duration, args.silence, args.stop_word)

    # Create job envelope for this stream
    job = JobEnvelope.create()
    job.config["stream"] = True
    job.config["input_device_ids"] = [d.id for d in input_devices]
    job.config["input_device_names"] = [d.name for d in input_devices]
    job.config["loopback_device_ids"] = [d.id for d in loopback_devices]
    job.config["loopback_device_names"] = [d.name for d in loopback_devices]
    job.config["language"] = args.language
    if args.duration:
        job.config["duration"] = args.duration
    if args.silence:
        job.config["silence"] = args.silence
    if args.stop_word:
        job.config["stop_word"] = args.stop_word

    run_dir = get_run_dir(job.id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # For now, whisper-stream only supports single device
    # If we have multiple devices, we need to record with ffmpeg first, then transcribe
    # If we have a single input device, we can use whisper-stream directly
    use_ffmpeg_recording = len(all_devices) > 1 or len(loopback_devices) > 0

    if use_ffmpeg_recording:
        # Record with ffmpeg (supports multiple devices and mixing)
        audio_path = run_dir / "audio.wav"
        device_names = [d.name for d in all_devices]
        logger.info(
            "Recording audio with ffmpeg from: %s", " + ".join(device_names)
        )
        # Check if verbose mode is enabled (DEBUG log level)
        import logging
        is_verbose = logger.isEnabledFor(logging.DEBUG)

        try:
            record_process = record_audio(
                input_devices=input_devices,
                loopback_devices=loopback_devices,
                output_path=audio_path,
                duration=args.duration,
                verbose=is_verbose,
            )
        except RuntimeError as e:
            logger.error("Failed to start recording: %s", e)
            job.update_status(JobStatus.FAILED, str(e))
            save_job(job)
            return 1

        # Monitor recording process
        try:
            return_code = _monitor_recording(
                record_process,
                duration=args.duration,
                silence=args.silence,
                stop_word=args.stop_word,
            )
        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")
            # Send SIGINT to ffmpeg to allow it to finish writing
            if record_process.poll() is None:
                record_process.send_signal(signal.SIGINT)
                try:
                    record_process.wait(timeout=3)
                    logger.info("ffmpeg stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("ffmpeg did not stop, killing")
                    record_process.kill()
                    record_process.wait()
            
            # Check if file was created (even if interrupted)
            if audio_path.exists() and audio_path.stat().st_size > 0:
                logger.info("Recording saved: %s (%d bytes)", audio_path.name, audio_path.stat().st_size)
                job.artifacts.append(str(audio_path))
                job.update_status(JobStatus.INTERRUPTED, "Recording interrupted")
            else:
                job.update_status(JobStatus.FAILED, "Recording interrupted - no file created")
            save_job(job)
            return 130  # Standard exit code for Ctrl+C

        # Wait for recording to finish
        record_process.wait()
        return_code = record_process.returncode

        # Read stderr before checking file (in case we need error messages)
        stderr = ""
        if record_process.stderr:
            try:
                # Read all available stderr
                stderr_bytes = record_process.stderr.read()
                if stderr_bytes:
                    stderr = stderr_bytes.decode("utf-8", errors="replace")
            except Exception as e:
                logger.debug("Could not read stderr: %s", e)

        # Check if file was created even if process failed (might be interrupted)
        file_created = audio_path.exists() and audio_path.stat().st_size > 0

        if return_code != 0:
            # Exit code 255 often means killed/interrupted, but file might still be valid
            if return_code == 255 and file_created:
                logger.warning(
                    "ffmpeg was interrupted (exit code 255) but file was created: %s",
                    audio_path.name,
                )
                # Continue processing - file is valid
            else:
                logger.error(
                    "ffmpeg recording failed (exit code %d)",
                    return_code,
                )
                
                # Show helpful error message (only if we captured stderr)
                # If verbose mode, stderr was already shown in real-time
                if not is_verbose and stderr:
                    # Extract error lines (usually contain "Error" or important info)
                    error_lines = [
                        line for line in stderr.split("\n")
                        if "error" in line.lower() or "Error" in line or "failed" in line.lower()
                    ]
                    if error_lines:
                        logger.error("ffmpeg errors:\n%s", "\n".join(error_lines[-5:]))
                    else:
                        # Show last few lines if no obvious errors
                        all_lines = stderr.strip().split("\n")
                        if all_lines:
                            logger.error("ffmpeg output (last 10 lines):\n%s", "\n".join(all_lines[-10:]))
                elif not is_verbose:
                    logger.error("No error output from ffmpeg (check if devices are accessible)")
                    logger.info("Run with -v or --verbose to see real-time ffmpeg output")
                
                if not file_created:
                    job.update_status(
                        JobStatus.FAILED, f"ffmpeg exited {return_code}"
                    )
                    save_job(job)
                    return 1

        # Verify output file was created (if we haven't already checked)
        if not file_created:
            if not audio_path.exists():
                logger.error(
                    "Recording completed but output file not found: %s", audio_path
                )
                job.update_status(JobStatus.FAILED, "Output file not created")
                save_job(job)
                return 1

            file_size = audio_path.stat().st_size
            if file_size == 0:
                logger.error("Recording created empty file: %s", audio_path)
                job.update_status(JobStatus.FAILED, "Empty output file")
                save_job(job)
                return 1
        else:
            file_size = audio_path.stat().st_size

        logger.info("Recording complete: %s (%d bytes)", audio_path.name, file_size)

        # Now transcribe the recorded audio
        audio_file = audio_path
        job.artifacts.append(str(audio_file))

        # Update job with input info
        job.input = InputFile.from_path(audio_file)
        job.update_status(JobStatus.RUNNING)
        save_job(job)

        if pipeline:
            # Run the specified pipeline on the captured audio
            logger.info("Running pipeline '%s' on captured audio...", pipeline.name)

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
            transcript_result = _run_whisper_cli_timestamps(
                audio_file=audio_file,
                model_path=model_path,
                language=args.language,
                run_dir=run_dir,
            )
            if transcript_result:
                for artifact in transcript_result:
                    job.artifacts.append(str(artifact))
                    logger.info("Created: %s", artifact.name)

    else:
        # Single input device: use whisper-stream (original behavior)
        device = input_devices[0]
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


def interactive_device_selection() -> tuple[list[AudioDevice], list[AudioDevice]]:
    """
    Interactive device picker with live audio meters.

    Shows devices with real-time audio level indicators and allows
    selecting multiple inputs and outputs. Continuously updates meters
    while waiting for user input.

    Returns:
        Tuple of (input_devices, loopback_devices).

    Raises:
        KeyboardInterrupt: If user cancels.
    """
    inputs = list_input_devices()
    outputs = list_output_devices()
    loopbacks = list_loopback_devices()

    if not inputs and not outputs:
        print("No audio devices found.", file=sys.stderr)
        raise ValueError("No devices available")

    # Show initial display
    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("Interactive Device Selection", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)
    print(
        "Watch the audio meters to see which devices are active.",
        file=sys.stderr,
    )
    print("Meters update every 0.5 seconds.", file=sys.stderr)
    print(file=sys.stderr)

    # Show meters continuously until user is ready
    selected_input_ids: set[int] = set()
    input_line = ""

    # First, show meters a few times so user can see activity
    for _ in range(3):
        _display_devices_with_meters(inputs, outputs, selected_input_ids)
        print(file=sys.stderr)
        print("Watching audio levels...", file=sys.stderr)
        time.sleep(0.5)

    # Now prompt for input
    default_input = str(inputs[0].id) if inputs else "-1"
    print(file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    prompt = (
        f"Select inputs (comma-separated IDs, -1 for none) [{default_input}]: "
    )

    try:
        input_line = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt

    print(file=sys.stderr)

    # Parse input selection
    if not input_line:
        input_line = default_input

    if input_line.strip() != "-1":
        for part in input_line.split(","):
            part = part.strip()
            if part:
                try:
                    device_id = int(part)
                    if any(d.id == device_id for d in inputs):
                        selected_input_ids.add(device_id)
                except ValueError:
                    pass

    # Show meters for output selection
    print(file=sys.stderr)
    print("Now select output devices (for system audio capture):", file=sys.stderr)
    print(file=sys.stderr)

    # Show meters a few times
    selected_loopback_ids: set[int] = set()
    for _ in range(3):
        _display_devices_with_meters(
            inputs, outputs, selected_input_ids, selected_loopback_ids
        )
        print(file=sys.stderr)
        print("Watching audio levels...", file=sys.stderr)
        time.sleep(0.5)

    # Prompt for output selection
    default_loopback = str(outputs[0].id) if outputs else "-1"
    print(file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    prompt = (
        f"Select outputs (comma-separated IDs, -1 for none) [{default_loopback}]: "
    )

    try:
        loopback_line = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt

    print(file=sys.stderr)

    # Parse loopback selection
    if not loopback_line:
        loopback_line = default_loopback

    if loopback_line.strip() != "-1":
        for part in loopback_line.split(","):
            part = part.strip()
            if part:
                try:
                    device_id = int(part)
                    if any(d.id == device_id for d in outputs):
                        selected_loopback_ids.add(device_id)
                except ValueError:
                    pass

    # Build result lists
    selected_inputs = [d for d in inputs if d.id in selected_input_ids]
    selected_loopbacks = [d for d in outputs if d.id in selected_loopback_ids]

    # Default to first device if nothing selected
    if not selected_inputs and not selected_loopbacks:
        if inputs:
            selected_inputs = [inputs[0]]
        if outputs:
            selected_loopbacks = [outputs[0]]

    return selected_inputs, selected_loopbacks


def _display_devices_with_meters(
    inputs: list[AudioDevice],
    outputs: list[AudioDevice],
    selected_input_ids: set[int] | None = None,
    selected_loopback_ids: set[int] | None = None,
    clear_screen: bool = True,
) -> None:
    """Display devices with live audio meters."""
    selected_input_ids = selected_input_ids or set()
    selected_loopback_ids = selected_loopback_ids or set()

    if clear_screen:
        print("\033[2J\033[H", end="", file=sys.stderr)  # Clear screen and move to top
    print("Available audio devices:", file=sys.stderr)
    print(file=sys.stderr)

    # Get audio levels for all devices
    all_devices = inputs + outputs
    levels = get_audio_levels(all_devices, duration=0.3)
    level_map = {d.id: levels.get(d.id, 0.0) for d in all_devices}

    if inputs:
        print("  INPUTS (can record from):", file=sys.stderr)
        for device in inputs:
            level = level_map.get(device.id, 0.0)
            meter = render_level_meter(level, width=5)
            selected = "✓" if device.id in selected_input_ids else " "
            name_padded = device.name.ljust(30)
            # Show capabilities
            caps = []
            if device.has_output:
                caps.append("output")
            if device.is_virtual:
                caps.append("virtual")
            cap_str = f" ({', '.join(caps)})" if caps else ""
            print(
                f"    [{device.id}] {name_padded} [{meter}] {selected}{cap_str}",
                file=sys.stderr,
            )
        print(file=sys.stderr)

    if outputs:
        print("  OUTPUTS (for system audio capture):", file=sys.stderr)
        for device in outputs:
            level = level_map.get(device.id, 0.0)
            meter = render_level_meter(level, width=5)
            selected = "✓" if device.id in selected_loopback_ids else " "
            name_padded = device.name.ljust(30)
            # Mark loopback devices
            loopback_marker = " [loopback]" if (device.is_virtual or device.direction == "loopback") else ""
            print(
                f"    [{device.id}] {name_padded} [{meter}] {selected}{loopback_marker}",
                file=sys.stderr,
            )
        print(file=sys.stderr)


def _monitor_recording(
    process: subprocess.Popen,
    duration: int | None,
    silence: int | None,
    stop_word: str | None,
) -> int:
    """
    Monitor a recording process for stop conditions.

    Args:
        process: Recording subprocess.
        duration: Max duration in seconds.
        silence: Stop after N seconds of silence.
        stop_word: Stop when phrase detected.

    Returns:
        Exit code (0 for success).

    Raises:
        KeyboardInterrupt: If user interrupts.
    """
    start_time = time.time()
    last_activity = time.time()

    try:
        while process.poll() is None:
            # Check duration
            if duration:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    logger.info("Stopping: duration limit (%ds)", duration)
                    break

            # Check silence (would need audio analysis - simplified for now)
            if silence:
                silent_for = time.time() - last_activity
                if silent_for >= silence:
                    logger.info("Stopping: silence (%ds)", silence)
                    break

            time.sleep(0.1)

    except KeyboardInterrupt:
        # Re-raise so caller can handle it
        raise

    # Stop the process gracefully
    if process.poll() is None:
        logger.debug("Sending SIGINT to ffmpeg process")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg did not stop gracefully, killing")
            process.kill()
            process.wait()

    return 0


def _list_devices() -> int:
    """
    List available audio devices with their capabilities.

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

    inputs = list_input_devices()
    outputs = list_output_devices()
    loopbacks = list_loopback_devices()

    print("Available audio devices:")
    print()
    if inputs:
        print("  INPUTS (can record from):")
        for device in inputs:
            caps = []
            if device.has_input:
                caps.append("input")
            if device.has_output:
                caps.append("output")
            if device.is_virtual:
                caps.append("virtual")
            cap_str = f" ({', '.join(caps)})" if caps else ""
            print(f"    [{device.id}] {device.name}{cap_str}")
        print()
    if outputs:
        print("  OUTPUTS (for system audio capture):")
        for device in outputs:
            markers = []
            if device.is_virtual or device.direction == "loopback":
                markers.append("loopback")
            if not device.has_input:
                markers.append("output-only")
            marker_str = f" [{', '.join(markers)}]" if markers else ""
            print(f"    [{device.id}] {device.name}{marker_str}")
        print()
    print("Use: infomux stream --input <id> --output <id>")
    print("Or:  infomux stream --prompt (interactive selection)")
    print()
    print("Note: To capture system audio, set a loopback device (BlackHole, etc.)")
    print("      as your system output in System Settings > Sound > Output")

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
        dest = run_dir / "audio_full.wav"
        src.rename(dest)
        return dest

    # Also check run_dir
    # Support both old and new names
    for pattern in ["audio_full.wav", "audio.wav", "output.wav", "*.wav"]:
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
