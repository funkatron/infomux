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


def _detect_speech_segments(ffmpeg: Path, audio_path: Path) -> list[dict]:
    """
    Detect speech segments in audio using silence detection.
    
    Uses ffmpeg silencedetect filter to find speech vs silence regions.
    
    Args:
        ffmpeg: Path to ffmpeg executable
        audio_path: Path to audio file
        
    Returns:
        List of dicts with start, end times of speech segments
    """
    # Get audio duration first
    import re
    duration_cmd = [
        str(ffmpeg),
        "-i", str(audio_path),
        "-f", "null",
        "-",
    ]
    duration_result = subprocess.run(
        duration_cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    # Parse duration from stderr
    duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", duration_result.stderr)
    if not duration_match:
        return []
    hours, minutes, seconds = map(float, duration_match.groups())
    total_duration = hours * 3600 + minutes * 60 + seconds
    
    # Use silencedetect to find silence regions
    # -af silencedetect=noise=-30dB:duration=0.3 finds silence below -30dB for 0.3s+
    cmd = [
        str(ffmpeg),
        "-i", str(audio_path),
        "-af", "silencedetect=noise=-30dB:duration=0.1",
        "-f", "null",
        "-",
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    
    # Parse silencedetect output
    # Format: "silence_start: 1.234" and "silence_end: 5.678 | silence_duration: 4.444"
    silence_regions = []
    
    for line in result.stderr.split("\n"):
        # Find silence_start
        start_match = re.search(r"silence_start: ([\d.]+)", line)
        if start_match:
            silence_regions.append({"start": float(start_match.group(1)), "end": None})
        
        # Find silence_end (paired with most recent start)
        end_match = re.search(r"silence_end: ([\d.]+)", line)
        if end_match and silence_regions:
            silence_regions[-1]["end"] = float(end_match.group(1))
    
    # Filter out incomplete silence regions
    silence_regions = [r for r in silence_regions if r["end"] is not None]
    
    # Build speech segments (inverse of silence regions)
    segments = []
    
    if not silence_regions:
        # No silence detected - entire audio is speech
        return [{"start": 0.0, "end": total_duration}]
    
    # First segment: 0 to first silence_start (if there's speech before first silence)
    if silence_regions[0]["start"] > 0.1:
        segments.append({"start": 0.0, "end": silence_regions[0]["start"]})
    
    # Middle segments: between silence_end and next silence_start
    for i in range(len(silence_regions) - 1):
        seg_start = silence_regions[i]["end"]
        seg_end = silence_regions[i + 1]["start"]
        if seg_end > seg_start:  # Only add if valid
            segments.append({"start": seg_start, "end": seg_end})
    
    # Last segment: last silence_end to end (if there's speech after last silence)
    if silence_regions:
        last_silence_end = silence_regions[-1]["end"]
        if last_silence_end < total_duration - 0.1:
            segments.append({"start": last_silence_end, "end": total_duration})
    
    return segments


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
    
    # Use ffmpeg's volumedetect to get average volume
    cmd = [
        str(ffmpeg),
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(audio_path),
        "-af", "volumedetect",
        "-f", "null",
        "-",
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    
    energy_info = {
        "has_audio": result.returncode == 0,
        "duration": duration,
    }
    
    # Parse mean_volume from volumedetect output
    # Format: "mean_volume: -20.5 dB"
    import re
    mean_match = re.search(r"mean_volume: ([\d.-]+) dB", result.stderr)
    if mean_match:
        energy_info["mean_volume_db"] = float(mean_match.group(1))
        # Consider > -50dB as likely speech
        energy_info["likely_speech"] = float(mean_match.group(1)) > -50.0
    
    max_match = re.search(r"max_volume: ([\d.-]+) dB", result.stderr)
    if max_match:
        energy_info["max_volume_db"] = float(max_match.group(1))
    
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
    audio_path = run_dir / "audio_full.wav"
    if not audio_path.exists():
        logger.warning("audio_full.wav not found, skipping audio analysis")
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
    
    # Detect actual speech segments from audio
    actual_speech_segments = []
    if audio_path:
        logger.info("detecting actual speech segments from audio...")
        actual_speech_segments = _detect_speech_segments(tools.ffmpeg, audio_path)
        logger.info("detected %d speech segments from audio", len(actual_speech_segments))
        for seg in actual_speech_segments[:5]:  # Show first 5
            logger.info("  speech segment: %.2f-%.2fs (duration: %.2fs)", seg["start"], seg["end"], seg["end"] - seg["start"])
    
    # Analyze audio energy if requested
    if args.audio_analysis and audio_path:
        logger.info("analyzing audio energy at word boundaries...")
        
        for i, word in enumerate(sample_words):
            energy = _analyze_audio_energy(
                tools.ffmpeg,
                audio_path,
                word["start_sec"],
                word["end_sec"],
            )
            
            # Check if word timing overlaps with actual speech
            overlaps_speech = False
            if actual_speech_segments:
                for seg in actual_speech_segments:
                    # Check if word timing overlaps with speech segment
                    if not (word["end_sec"] < seg["start"] or word["start_sec"] > seg["end"]):
                        overlaps_speech = True
                        break
            
            logger.info(
                "word %d '%s' (%.2f-%.2fs): energy=%s, overlaps_speech=%s",
                i + 1,
                word["text"],
                word["start_sec"],
                word["end_sec"],
                energy.get("mean_volume_db", "unknown"),
                overlaps_speech,
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
    
    # Compare expected vs actual speech timing
    if actual_speech_segments:
        logger.info("")
        logger.info("Timing Comparison:")
        logger.info("Expected word timings vs detected speech segments:")
        for i, word in enumerate(sample_words):
            # Find closest speech segment
            closest_seg = None
            min_distance = float("inf")
            for seg in actual_speech_segments:
                # Distance from word start to segment start
                dist = abs(word["start_sec"] - seg["start"])
                if dist < min_distance:
                    min_distance = dist
                    closest_seg = seg
            
            if closest_seg:
                offset = word["start_sec"] - closest_seg["start"]
                logger.info(
                    "  word %d '%s': expected %.2fs, actual speech starts %.2fs (offset: %+.2fs)",
                    i + 1,
                    word["text"],
                    word["start_sec"],
                    closest_seg["start"],
                    offset,
                )
    
    return 0
