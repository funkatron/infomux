"""
Audio device discovery and recording for infomux.

Uses ffmpeg's avfoundation (macOS) to discover and record from audio devices.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

from infomux.config import find_tool
from infomux.log import get_logger

logger = get_logger(__name__)


@dataclass
class AudioDevice:
    """
    Represents an audio input device.

    Attributes:
        id: Device index (used for selection)
        name: Human-readable device name
        device_type: 'audio' or 'video'
    """

    id: int
    name: str
    device_type: str = "audio"


def list_audio_devices() -> list[AudioDevice]:
    """
    List available audio input devices.

    Uses ffmpeg's avfoundation to enumerate devices on macOS.

    Returns:
        List of AudioDevice objects.

    Raises:
        RuntimeError: If ffmpeg is not found or device listing fails.
    """
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    # Run ffmpeg to list devices
    # This outputs to stderr and "fails" with exit code, but that's expected
    result = subprocess.run(
        [str(ffmpeg), "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )

    # Parse the output (in stderr)
    output = result.stderr
    devices = []

    # Look for audio devices section
    # Lines look like: [AVFoundation indev @ 0x...] [0] Device Name
    in_audio_section = False
    device_pattern = re.compile(r"\]\s+\[(\d+)\]\s+(.+)")

    for line in output.split("\n"):
        if "AVFoundation audio devices:" in line:
            in_audio_section = True
            continue
        if in_audio_section:
            # Stop if we hit an error line
            if "Error" in line or "error" in line.lower():
                break

            # Parse device line
            match = device_pattern.search(line)
            if match:
                device_id = int(match.group(1))
                device_name = match.group(2).strip()
                devices.append(AudioDevice(id=device_id, name=device_name))

    return devices


def select_audio_device(devices: list[AudioDevice]) -> AudioDevice | None:
    """
    Interactively prompt user to select an audio device.

    Args:
        devices: List of available devices.

    Returns:
        Selected AudioDevice, or None if cancelled.
    """
    if not devices:
        print("No audio devices found.", file=sys.stderr)
        return None

    print("\nAvailable audio devices:", file=sys.stderr)
    for device in devices:
        print(f"  [{device.id}] {device.name}", file=sys.stderr)
    print(file=sys.stderr)

    # Default to first device
    default_id = devices[0].id
    prompt = f"Select device [{default_id}]: "

    try:
        user_input = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return None

    # Parse selection
    if not user_input:
        selected_id = default_id
    else:
        try:
            selected_id = int(user_input)
        except ValueError:
            print(f"Invalid selection: {user_input}", file=sys.stderr)
            return None

    # Find the device
    for device in devices:
        if device.id == selected_id:
            return device

    print(f"Device {selected_id} not found.", file=sys.stderr)
    return None


def get_device_by_id(device_id: int) -> AudioDevice | None:
    """
    Get a device by its ID.

    Args:
        device_id: Device index.

    Returns:
        AudioDevice if found, None otherwise.
    """
    devices = list_audio_devices()
    for device in devices:
        if device.id == device_id:
            return device
    return None
