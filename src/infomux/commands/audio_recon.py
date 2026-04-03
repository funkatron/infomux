"""
Audio recon: quick check that system audio reaches a recordable loopback path.

One command: record a short sample, measure levels, exit 0 if not silent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from infomux.audio import (
    choose_recon_capture_device,
    get_output_device_by_id,
    list_output_devices,
    record_audio,
)
from infomux.config import find_tool
from infomux.job import JobEnvelope
from infomux.log import get_logger
from infomux.storage import get_run_dir

logger = get_logger(__name__)

EXIT_PASS = 0
EXIT_ERROR = 1
EXIT_SILENT = 2


def configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--duration",
        type=int,
        default=8,
        metavar="SEC",
        help="Recording length in seconds (default: 8).",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-80.0,
        metavar="DB",
        help="max_volume at or below this dB counts as silent (default: -80).",
    )
    parser.add_argument(
        "--output-id",
        type=int,
        default=None,
        help="Force capture device ID from `infomux stream --list-devices` OUTPUTS.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Force capture device by substring match in OUTPUTS (overrides auto-pick).",
    )
    parser.add_argument(
        "--switch-output",
        type=str,
        default=None,
        metavar="NAME",
        help="Set system playback device via SwitchAudioSource before capture "
        "(e.g. Multi-Output aggregate name).",
    )
    parser.add_argument(
        "--sleep-after-switch",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds to sleep after switch (default: INFOMUX_RECON_SLEEP_AFTER_SWITCH or 0).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Resolve and print chosen capture device, then exit (no recording).",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the captured WAV with afplay (or ffplay) after recording.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object with result to stdout (still logs to stderr).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less stderr noise (still prints errors).",
    )


def _find_switch_audio_source() -> str | None:
    p = shutil.which("SwitchAudioSource")
    if p:
        return p
    for path in ("/opt/homebrew/bin/SwitchAudioSource", "/usr/local/bin/SwitchAudioSource"):
        if Path(path).is_file():
            return path
    return None


def _resolve_capture_device(args: Namespace):
    """Return AudioDevice or None."""
    if args.output_id is not None:
        return get_output_device_by_id(args.output_id)
    if args.output_name:
        name = args.output_name.strip()
        for d in list_output_devices():
            if not d.has_input:
                continue
            if name.lower() in d.name.lower() or d.name == name:
                return d
        return None
    return choose_recon_capture_device()


def _wav_peak_db(wav_path: Path) -> tuple[float | None, float | None]:
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        return None, None
    r = subprocess.run(
        [str(ffmpeg), "-i", str(wav_path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    err = r.stderr
    max_m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", err)
    mean_m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", err)
    max_db = float(max_m.group(1)) if max_m else None
    mean_db = float(mean_m.group(1)) if mean_m else None
    return max_db, mean_db


def _play_wav(path: Path) -> None:
    if shutil.which("afplay"):
        subprocess.run(["afplay", str(path)], check=False)
    elif shutil.which("ffplay"):
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
            check=False,
        )
    else:
        logger.warning("afplay/ffplay not found; skipping --play")


def execute(args: Namespace) -> int:
    quiet = getattr(args, "quiet", False)
    duration = max(1, int(args.duration))
    threshold = float(args.silence_threshold)

    sleep_after = args.sleep_after_switch
    if sleep_after is None:
        raw = os.environ.get("INFOMUX_RECON_SLEEP_AFTER_SWITCH", "0").strip()
        try:
            sleep_after = float(raw)
        except ValueError:
            sleep_after = 0.0

    if args.switch_output:
        sacmd = _find_switch_audio_source()
        if sacmd:
            if not quiet:
                logger.info("switching system output to %r via %s", args.switch_output, sacmd)
            r = subprocess.run([sacmd, "-s", args.switch_output], capture_output=True, text=True)
            if r.returncode != 0:
                logger.error("SwitchAudioSource failed: %s", r.stderr or r.stdout)
                return EXIT_ERROR
            if sleep_after > 0:
                if not quiet:
                    logger.info("sleeping %.1fs after output switch", sleep_after)
                time.sleep(sleep_after)
        else:
            logger.warning(
                "SwitchAudioSource not found; install: brew install switchaudio-osx"
            )

    try:
        device = _resolve_capture_device(args)
    except Exception as e:
        logger.error("device resolution failed: %s", e)
        return EXIT_ERROR

    if device is None:
        logger.error(
            "no suitable capture device (need OUTPUT with input). "
            "Set INFOMUX_RECON_CAPTURE or use --output-id / --output-name."
        )
        return EXIT_ERROR

    if args.check_only:
        if not args.json:
            print("Chosen capture device:", device.name, f"(id={device.id})", file=sys.stderr)
        else:
            print(
                json.dumps(
                    {
                        "device_name": device.name,
                        "device_id": device.id,
                        "check_only": True,
                    }
                )
            )
        return EXIT_PASS

    job = JobEnvelope.create()
    run_dir = get_run_dir(job.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    audio_path = run_dir / "audio.wav"

    if not quiet:
        logger.info(
            "recording %ds from %s (id=%s) -> %s",
            duration,
            device.name,
            device.id,
            audio_path,
        )

    try:
        proc = record_audio(
            input_devices=[],
            loopback_devices=[device],
            output_path=audio_path,
            duration=duration,
            verbose=False,
        )
    except RuntimeError as e:
        logger.error("%s", e)
        return EXIT_ERROR

    try:
        code = proc.wait(timeout=duration + 30)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        code = proc.returncode

    if code not in (0, 255) and not (audio_path.exists() and audio_path.stat().st_size > 0):
        logger.error("recording failed (exit %s)", code)
        return EXIT_ERROR

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        logger.error("no audio file produced")
        return EXIT_ERROR

    if args.play:
        if not quiet:
            logger.info("playing %s", audio_path)
        _play_wav(audio_path)

    max_db, mean_db = _wav_peak_db(audio_path)
    if max_db is None:
        logger.error("could not parse levels from ffmpeg")
        return EXIT_ERROR

    silent = max_db <= threshold
    result = {
        "pass": not silent,
        "silent": silent,
        "max_volume_db": max_db,
        "mean_volume_db": mean_db,
        "silence_threshold_db": threshold,
        "device_name": device.name,
        "device_id": device.id,
        "duration_sec": duration,
        "audio_wav": str(audio_path),
        "run_dir": str(run_dir),
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(f"mean_volume: {mean_db} dB", file=sys.stderr)
        print(f"max_volume:  {max_db} dB", file=sys.stderr)
        print(f"silence_threshold: {threshold} dB", file=sys.stderr)
        if silent:
            print(
                "FAIL: capture appears silent (max_volume <= threshold).",
                file=sys.stderr,
            )
            print(
                "Route system audio through BlackHole or a Multi-Output / aggregate device.",
                file=sys.stderr,
            )
        else:
            print("PASS: non-silent audio captured.", file=sys.stderr)
        print(run_dir, file=sys.stdout)

    return EXIT_SILENT if silent else EXIT_PASS
