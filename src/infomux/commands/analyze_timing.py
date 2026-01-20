"""
Analyze timing accuracy of lyric videos by comparing video frames to audio.

Extracts frames at word timestamps and analyzes audio to verify word timing.
"""

from __future__ import annotations

import json
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path

from infomux.config import get_tool_paths
from infomux.log import get_logger
from infomux.storage import get_run_dir

logger = get_logger(__name__)


def configure_parser(parser: ArgumentParser) -> None:
    """Configure argument parser for analyze-timing command."""
    parser.add_argument(
        "run_id",
        help="Run ID to analyze (e.g., run-20260120-220733-0cf45b)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of sample frames to extract for visual inspection (default: 10)",
    )
    parser.add_argument(
        "--audio-analysis",
        action="store_true",
        help="Analyze audio to detect actual word boundaries using energy/silence detection",
    )


def _extract_frame_at_time(
    ffmpeg: Path, video_path: Path, timestamp: float, output_path: Path
) -> bool:
    """
    Extract a single frame from video at specific timestamp.
    
    Args:
        ffmpeg: Path to ffmpeg executable
        video_path: Path to video file
        timestamp: Timestamp in seconds
        output_path: Path to save frame image
        
    Returns:
        True if successful
    """
    cmd = [
        str(ffmpeg),
        "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",  # High quality
        str(output_path),
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    
    return result.returncode == 0 and output_path.exists()


def _analyze_audio_energy(
    ffmpeg: Path, audio_path: Path, start_time: float, end_time: float
) -> dict:
    """
    Analyze audio energy in a time range to detect if speech is present.
    
    Args:
        ffmpeg: Path to ffmpeg executable
        audio_path: Path to audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Dict with energy analysis results
    """
    duration = end_time - start_time
    
    # Use ffmpeg's astats filter to analyze audio
    cmd = [
        str(ffmpeg),
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(audio_path),
        "-af", "astats=metadata=1:reset=1",
        "-f", "null",
        "-",
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    
    # Parse energy from astats output
    # This is a simplified version - could be enhanced
    energy_info = {
        "has_audio": result.returncode == 0,
        "duration": duration,
    }
    
    # Try to extract RMS or peak levels from stderr
    if "RMS level" in result.stderr:
        # Parse RMS level (e.g., "RMS level: -20.1 dB")
        import re
        rms_match = re.search(r"RMS level: ([\d.-]+) dB", result.stderr)
        if rms_match:
            energy_info["rms_db"] = float(rms_match.group(1))
    
    return energy_info


def _parse_transcript_json(transcript_path: Path) -> list[dict]:
    """
    Parse transcript.json and extract word timings.
    
    Returns:
        List of word dicts with text, start_ms, end_ms
    """
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)
    
    words = []
    for segment in data.get("transcription", []):
        for token in segment.get("tokens", []):
            timestamps = token.get("timestamps", {})
            if "from" in timestamps and "to" in timestamps:
                # Parse timestamp format HH:MM:SS,mmm
                def parse_timestamp(ts: str) -> int:
                    parts = ts.replace(",", ".").split(":")
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = float(parts[2])
                    return int((hours * 3600 + minutes * 60 + seconds) * 1000)
                
                start_ms = parse_timestamp(timestamps["from"])
                end_ms = parse_timestamp(timestamps["to"])
                
                words.append({
                    "text": token.get("text", "").strip(),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "start_sec": start_ms / 1000.0,
                    "end_sec": end_ms / 1000.0,
                })
    
    return words


def execute(args: Namespace) -> int:
    """
    Execute the analyze-timing command.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    run_dir = get_run_dir(args.run_id)
    if not run_dir.exists():
        logger.error("run not found: %s", args.run_id)
        return 1
    
    tools = get_tool_paths()
    if not tools.ffmpeg:
        logger.error("ffmpeg not found")
        return 1
    
    # Find video file
    video_files = list(run_dir.glob("*_lyric_video.mp4"))
    if not video_files:
        logger.error("no lyric video found in run directory")
        return 1
    
    video_path = video_files[0]
    logger.info("analyzing video: %s", video_path.name)
    
    # Find transcript.json
    transcript_path = run_dir / "transcript.json"
    if not transcript_path.exists():
        logger.error("transcript.json not found")
        return 1
    
    # Parse transcript
    words = _parse_transcript_json(transcript_path)
    if not words:
        logger.error("no words found in transcript.json")
        return 1
    
    logger.info("found %d words in transcript", len(words))
    
    # Find audio file for analysis
    audio_path = run_dir / "audio.wav"
    if not audio_path.exists():
        logger.warning("audio.wav not found, skipping audio analysis")
        audio_path = None
    
    # Create frames directory
    frames_dir = run_dir / "timing_analysis_frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Extract sample frames at word timestamps
    logger.info("extracting %d sample frames...", args.frames)
    sample_words = words[:args.frames] if len(words) >= args.frames else words
    
    extracted_frames = []
    for i, word in enumerate(sample_words):
        # Extract frame at word start time
        frame_path = frames_dir / f"word_{i+1:03d}_{word['start_sec']:.2f}s.png"
        
        if _extract_frame_at_time(
            tools.ffmpeg, video_path, word["start_sec"], frame_path
        ):
            extracted_frames.append({
                "word": word["text"],
                "expected_time": word["start_sec"],
                "frame_path": frame_path,
            })
            logger.info(
                "extracted frame %d: '%s' at %.2fs -> %s",
                i + 1,
                word["text"],
                word["start_sec"],
                frame_path.name,
            )
        else:
            logger.warning("failed to extract frame for word '%s' at %.2fs", word["text"], word["start_sec"])
    
    # Analyze audio if requested
    if args.audio_analysis and audio_path:
        logger.info("analyzing audio energy at word boundaries...")
        
        for i, word in enumerate(sample_words):
            energy = _analyze_audio_energy(
                tools.ffmpeg,
                audio_path,
                word["start_sec"],
                word["end_sec"],
            )
            
            logger.info(
                "word %d '%s' (%.2f-%.2fs): energy=%s",
                i + 1,
                word["text"],
                word["start_sec"],
                word["end_sec"],
                energy.get("rms_db", "unknown"),
            )
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Timing Analysis Summary")
    logger.info("=" * 60)
    logger.info("Video: %s", video_path.name)
    logger.info("Total words: %d", len(words))
    logger.info("Sample frames extracted: %d", len(extracted_frames))
    logger.info("Frames directory: %s", frames_dir)
    logger.info("")
    logger.info("First %d words:", len(sample_words))
    for i, word in enumerate(sample_words):
        logger.info(
            "  %d. '%s' at %.2f-%.2fs (duration: %.2fs)",
            i + 1,
            word["text"],
            word["start_sec"],
            word["end_sec"],
            word["end_sec"] - word["start_sec"],
        )
    logger.info("")
    logger.info("View extracted frames in: %s", frames_dir)
    logger.info("Compare frame content to expected word timing above")
    
    return 0
