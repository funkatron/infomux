"""
Embed subtitles into video as a soft-coded stream.

Uses ffmpeg to remux subtitles into the video container without re-encoding.
Subtitles can be toggled on/off by the viewer in most players.

For burn-in (hardcoded subtitles), use the separate burn_subs step.

Command (soft subs):
    ffmpeg -i video.mp4 -i subs.srt -c copy -c:s mov_text output.mp4

Command (burn-in):
    ffmpeg -i video.mp4 -vf "subtitles=subs.srt" -c:a copy output.mp4
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from infomux.config import get_tool_paths
from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step

logger = get_logger(__name__)


@register_step
@dataclass
class EmbedSubsStep:
    """
    Pipeline step to embed subtitles into a video file.

    By default, creates soft-coded subtitles that viewers can toggle.
    Set burn_in=True to permanently render subtitles into the video.

    When used in a pipeline, this step:
    - Takes the original video as input (input_from=None)
    - Finds the .srt file in the output directory (from transcribe_timed)
    """

    name: str = "embed_subs"
    burn_in: bool = False
    language: str = "eng"
    subtitle_style: dict = field(default_factory=dict)

    def execute(
        self,
        input_path: Path,
        output_dir: Path,
    ) -> list[Path]:
        """
        Embed subtitles into video.

        In pipeline mode, input_path is the original video, and we find
        the subtitle file (transcript.srt) in output_dir.
        """
        # Find subtitle file in output_dir
        srt_path = output_dir / "transcript.srt"
        if srt_path.exists():
            subtitle_path = srt_path
        else:
            # Look for any .srt file
            srt_files = list(output_dir.glob("*.srt"))
            if srt_files:
                subtitle_path = srt_files[0]
            else:
                raise StepError(self.name, "No subtitle file found in output directory")

        return self._embed(input_path, subtitle_path, output_dir)

    def _embed(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_dir: Path,
    ) -> list[Path]:
        """
        Embed subtitles into video using ffmpeg.

        Args:
            video_path: Path to the original video file.
            subtitle_path: Path to the subtitle file (SRT or VTT).
            output_dir: Directory to write the output video.

        Returns:
            List containing the path to the output video.

        Raises:
            StepError: If ffmpeg fails or files not found, or if input is audio-only.
        """
        tools = get_tool_paths()

        if not tools.ffmpeg:
            raise StepError(self.name, "ffmpeg not found")

        if not video_path.exists():
            raise StepError(self.name, f"video not found: {video_path}")

        if not subtitle_path.exists():
            raise StepError(self.name, f"subtitles not found: {subtitle_path}")

        # Check if input is audio-only (not a video file)
        audio_extensions = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".opus"}
        if video_path.suffix.lower() in audio_extensions:
            raise StepError(
                self.name,
                f"Input file is audio-only ({video_path.suffix}), but embed_subs requires a video file. "
                f"Use the 'audio-to-video' pipeline instead to generate a video from audio with subtitles: "
                f"infomux run --pipeline audio-to-video {video_path}"
            )

        # Output filename
        suffix = "_captioned" if not self.burn_in else "_burned"
        output_path = output_dir / f"{video_path.stem}{suffix}{video_path.suffix}"

        logger.info(
            "embedding subtitles (%s): %s",
            "burn-in" if self.burn_in else "soft",
            video_path.name,
        )

        if self.burn_in:
            cmd = self._build_burn_in_cmd(
                tools.ffmpeg, video_path, subtitle_path, output_path
            )
        else:
            cmd = self._build_soft_sub_cmd(
                tools.ffmpeg, video_path, subtitle_path, output_path
            )

        logger.debug("running: %s", " ".join(str(c) for c in cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                logger.error("ffmpeg stderr: %s", result.stderr[-500:])
                raise StepError(
                    self.name,
                    f"ffmpeg failed: {result.returncode}",
                )

            if not output_path.exists():
                raise StepError(self.name, f"output not created: {output_path}")

            output_size = output_path.stat().st_size
            logger.info(
                "created: %s (%.1f MB)",
                output_path.name,
                output_size / 1024 / 1024,
            )

            return [output_path]

        except FileNotFoundError:
            raise StepError(self.name, f"ffmpeg not found: {tools.ffmpeg}")
        except subprocess.SubprocessError as e:
            raise StepError(self.name, f"subprocess error: {e}")

    def _build_soft_sub_cmd(
        self,
        ffmpeg: Path,
        video: Path,
        subs: Path,
        output: Path,
    ) -> list:
        """Build ffmpeg command for soft-coded subtitles."""
        return [
            str(ffmpeg),
            "-y",  # Overwrite output
            "-i", str(video),
            "-i", str(subs),
            "-c", "copy",  # Copy video/audio streams
            "-c:s", "mov_text",  # Subtitle codec for MP4
            "-metadata:s:s:0", f"language={self.language}",
            str(output),
        ]

    def _build_burn_in_cmd(
        self,
        ffmpeg: Path,
        video: Path,
        subs: Path,
        output: Path,
    ) -> list:
        """Build ffmpeg command for burned-in subtitles."""
        # Escape path for ffmpeg filter
        subs_escaped = str(subs).replace(":", "\\:").replace("'", "\\'")

        # Build subtitle filter with optional styling
        sub_filter = f"subtitles='{subs_escaped}'"

        if self.subtitle_style:
            # Add ASS-style formatting
            style_parts = []
            if "fontsize" in self.subtitle_style:
                style_parts.append(f"FontSize={self.subtitle_style['fontsize']}")
            if "fontname" in self.subtitle_style:
                style_parts.append(f"FontName={self.subtitle_style['fontname']}")
            if "primarycolor" in self.subtitle_style:
                style_parts.append(
                    f"PrimaryColour={self.subtitle_style['primarycolor']}"
                )

            if style_parts:
                sub_filter += f":force_style='{','.join(style_parts)}'"

        return [
            str(ffmpeg),
            "-y",
            "-i", str(video),
            "-vf", sub_filter,
            "-c:a", "copy",  # Copy audio stream
            str(output),
        ]


def run(
    input_path: Path,
    output_dir: Path,
    burn_in: bool = False,
) -> StepResult:
    """
    Run the embed_subs step.

    Args:
        input_path: Path to video file.
        output_dir: Directory for output (must contain transcript.srt).
        burn_in: If True, burn subtitles into video. If False, soft-code.

    Returns:
        StepResult with execution details.
    """
    step = EmbedSubsStep(burn_in=burn_in)
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
