"""
Helpers for preparing audio specifically for transcription.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


TRANSCRIPTION_AUDIO_FILENAME = "audio_for_transcription.wav"


def ensure_transcription_audio(
    input_path: Path,
    output_dir: Path,
    ffmpeg_path: Path,
) -> Path:
    """
    Convert an input audio file to the whisper-friendly mono/16kHz format.

    Always writes a separate artifact so capture audio can remain high-fidelity.
    """
    output_path = output_dir / TRANSCRIPTION_AUDIO_FILENAME

    cmd = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "ffmpeg conversion failed"
        raise RuntimeError(msg)
    if not output_path.exists():
        raise RuntimeError(f"transcription audio not created: {output_path}")

    return output_path
