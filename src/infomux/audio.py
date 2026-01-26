"""
Audio device discovery and recording for infomux.

Uses ffmpeg's avfoundation (macOS) to discover and record from audio devices.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from infomux.config import find_tool
from infomux.log import get_logger

logger = get_logger(__name__)

# Known loopback device patterns (virtual devices that capture system audio)
KNOWN_LOOPBACK_PATTERNS = [
    "BlackHole",
    "Screenflick Loopback",
    "Loopback Audio",
    "Soundflower",
]


@dataclass
class AudioDevice:
    """
    Represents an audio device with input and/or output capabilities.

    Attributes:
        id: Device index (used for selection)
        name: Human-readable device name
        device_type: 'audio' or 'video'
        direction: 'input' (microphone), 'loopback' (system audio capture), or 'both'
        has_input: Whether device can be used as input (microphone)
        has_output: Whether device can be used as output (speakers)
        is_virtual: Whether device is virtual (loopback device)
    """

    id: int
    name: str
    device_type: str = "audio"
    direction: str = field(default="input")
    has_input: bool = field(default=True)
    has_output: bool = field(default=False)
    is_virtual: bool = field(default=False)


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
                device = AudioDevice(id=device_id, name=device_name)
                device = classify_device(device)
                devices.append(device)

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


def get_device_capabilities() -> dict[str, dict[str, bool]]:
    """
    Query system_profiler to get device input/output capabilities.

    Returns:
        Dictionary mapping device name -> {'input': bool, 'output': bool, 'virtual': bool}
    """
    try:
        result = subprocess.run(
            ["system_profiler", "SPAudioDataType"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout

        devices: dict[str, dict[str, bool]] = {}
        current_device: str | None = None

        for line in output.split("\n"):
            # Device name appears as a line ending with colon, not indented
            # Format: "        Device Name:"
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this is a device name (ends with colon, might have leading spaces)
            if stripped.endswith(":") and "Devices" not in stripped:
                # Extract device name (remove colon and any leading/trailing whitespace)
                device_name = stripped.rstrip(":").strip()
                # Skip section headers
                if device_name and device_name not in ["Devices", "Audio"]:
                    current_device = device_name
                    if device_name not in devices:
                        devices[device_name] = {
                            "input": False,
                            "output": False,
                            "virtual": False,
                        }
            elif current_device and line.strip():
                # Check for capabilities in the device's properties
                stripped_lower = stripped.lower()
                if "input channels:" in stripped_lower:
                    devices[current_device]["input"] = True
                elif "output channels:" in stripped_lower:
                    devices[current_device]["output"] = True
                elif "transport:" in stripped_lower and "virtual" in stripped_lower:
                    devices[current_device]["virtual"] = True

        return devices
    except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        logger.debug("Could not query system_profiler for device capabilities: %s", e)
        return {}


def classify_device(device: AudioDevice) -> AudioDevice:
    """
    Detect device capabilities and classify appropriately.

    Uses system_profiler to determine if device has input/output capabilities,
    and checks known loopback patterns.

    Args:
        device: AudioDevice to classify.

    Returns:
        AudioDevice with capabilities and direction set appropriately.
    """
    # Get capabilities from system_profiler
    capabilities = get_device_capabilities()
    device_caps = capabilities.get(device.name, {})

    # Set capabilities
    device.has_input = device_caps.get("input", True)  # Default to True (most devices have input)
    device.has_output = device_caps.get("output", False)
    device.is_virtual = device_caps.get("virtual", False)

    # Classify direction
    device_name_lower = device.name.lower()

    # Check for known loopback patterns
    is_loopback = False
    for pattern in KNOWN_LOOPBACK_PATTERNS:
        if pattern.lower() in device_name_lower:
            is_loopback = True
            break

    # Virtual devices are typically loopbacks
    if device.is_virtual:
        is_loopback = True

    # Set direction
    if is_loopback:
        device.direction = "loopback"
    elif device.has_input and device.has_output:
        device.direction = "both"
    elif device.has_output and not device.has_input:
        device.direction = "output"
    else:
        device.direction = "input"

    return device


def list_input_devices() -> list[AudioDevice]:
    """
    Return devices that can be used as input (microphones).

    Returns:
        List of AudioDevice objects that have input capability.
    """
    all_devices = list_audio_devices()
    return [d for d in all_devices if d.has_input]


def list_output_devices() -> list[AudioDevice]:
    """
    Return devices that can be used as output (speakers/loopback).

    Includes both physical output devices and virtual loopback devices.
    Also includes output-only devices from system_profiler that aren't in ffmpeg's list.

    Returns:
        List of AudioDevice objects that have output capability or are loopback.
    """
    # Start with devices from ffmpeg
    all_devices = list_audio_devices()
    device_map = {d.name: d for d in all_devices}

    # Add output-only devices from system_profiler that ffmpeg doesn't see
    capabilities = get_device_capabilities()
    next_id = max([d.id for d in all_devices], default=-1) + 1

    for device_name, caps in capabilities.items():
        # Skip if already in ffmpeg list
        if device_name in device_map:
            continue

        # Add output-only devices (they can't be recorded from, but user should see them)
        if caps.get("output") and not caps.get("input"):
            device = AudioDevice(
                id=next_id,
                name=device_name,
                direction="output",
                has_input=False,
                has_output=True,
                is_virtual=caps.get("virtual", False),
            )
            all_devices.append(device)
            device_map[device_name] = device
            next_id += 1

    # Include devices with output capability OR loopback devices
    return [
        d
        for d in all_devices
        if d.has_output or d.direction == "loopback" or d.is_virtual
    ]


def list_loopback_devices() -> list[AudioDevice]:
    """
    Return only loopback devices (system audio capture).

    Returns:
        List of AudioDevice objects with direction='loopback' or is_virtual=True.
    """
    all_devices = list_audio_devices()
    return [d for d in all_devices if d.direction == "loopback" or d.is_virtual]


def get_default_input() -> AudioDevice | None:
    """
    Return first available input device, preferring non-virtual devices.

    Returns:
        AudioDevice if found, None otherwise.
    """
    inputs = list_input_devices()
    if not inputs:
        return None

    # Prefer non-virtual devices (actual microphones over loopback devices)
    non_virtual = [d for d in inputs if not d.is_virtual]
    if non_virtual:
        return non_virtual[0]

    # Fall back to virtual devices if that's all we have
    return inputs[0]


def get_default_output() -> AudioDevice | None:
    """
    Return first available output device (preferring loopback devices).

    Returns:
        AudioDevice if found, None otherwise.
    """
    # Prefer loopback devices for system audio capture
    loopbacks = list_loopback_devices()
    if loopbacks:
        return loopbacks[0]

    # Fall back to any output-capable device
    outputs = list_output_devices()
    return outputs[0] if outputs else None


def get_default_loopback() -> AudioDevice | None:
    """
    Return first available loopback device.

    Returns:
        AudioDevice if found, None otherwise.
    """
    loopbacks = list_loopback_devices()
    return loopbacks[0] if loopbacks else None


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


def get_audio_levels(
    devices: list[AudioDevice], duration: float = 0.5
) -> dict[int, float]:
    """
    Sample audio levels from devices using ffmpeg.

    Uses ffmpeg's volumedetect filter to measure audio levels.
    Returns normalized levels (0.0 = silence, 1.0 = max).

    Args:
        devices: List of devices to sample.
        duration: Sample duration in seconds (default: 0.5).

    Returns:
        Dictionary mapping device_id -> level (0.0-1.0).
    """
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        return {}

    levels: dict[int, float] = {}

    for device in devices:
        try:
            # Use ffmpeg to sample audio and detect volume
            # -t duration: sample for specified time
            # -af volumedetect: analyze volume levels
            # -f null: discard output (we only care about stderr stats)
            cmd = [
                str(ffmpeg),
                "-f", "avfoundation",
                "-i", f":{device.id}",
                "-t", str(duration),
                "-af", "volumedetect",
                "-f", "null",
                "-",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration + 1,
            )

            # Parse volumedetect output from stderr
            # Looks like: mean_volume: -XX.X dB
            output = result.stderr
            mean_match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", output)
            max_match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", output)

            if mean_match and max_match:
                # Convert dB to linear scale (0-1)
                # Typical range: -60dB (silence) to 0dB (max)
                max_db = float(max_match.group(1))

                # Normalize: -60dB = 0.0, 0dB = 1.0
                # Use max_volume as it's more representative of activity
                normalized = max(0.0, min(1.0, (max_db + 60.0) / 60.0))
                levels[device.id] = normalized
            else:
                # No audio detected or error
                levels[device.id] = 0.0

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
            logger.debug("Failed to sample audio level for device %d: %s", device.id, e)
            levels[device.id] = 0.0

    return levels


def render_level_meter(level: float, width: int = 5) -> str:
    """
    Render audio level as Unicode bar characters.

    Args:
        level: Audio level (0.0-1.0).
        width: Width of meter in characters (default: 5).

    Returns:
        String with bar characters, e.g., "[▁▂▃▅▇]"
    """
    bars = "▁▂▃▄▅▆▇█"
    num_bars = len(bars)

    # Clamp level to 0-1
    level = max(0.0, min(1.0, level))

    # Calculate how many bars to show
    filled = int(level * width * num_bars / num_bars)
    filled = min(filled, width)

    # Build meter string
    meter_chars = []
    for i in range(width):
        if i < filled:
            # Calculate which bar character to use
            bar_index = int((level * num_bars) * (i + 1) / width)
            bar_index = min(bar_index, num_bars - 1)
            meter_chars.append(bars[bar_index])
        else:
            meter_chars.append("▁")

    return "".join(meter_chars)


def record_audio(
    input_devices: list[AudioDevice],
    loopback_devices: list[AudioDevice],
    output_path: Path,
    duration: int | None = None,
    sample_rate: int = 16000,
    mono: bool = True,
    verbose: bool = False,
) -> subprocess.Popen:
    """
    Record and mix audio from multiple devices using ffmpeg.

    Args:
        input_devices: List of input devices (microphones).
        loopback_devices: List of loopback devices (system audio).
        output_path: Path to write output WAV file.
        duration: Maximum recording duration in seconds (None = unlimited).
        sample_rate: Output sample rate (default: 16000).
        mono: Whether to output mono audio (default: True).
        verbose: If True, stream ffmpeg stderr to terminal.

    Returns:
        subprocess.Popen object for the ffmpeg process.

    Raises:
        RuntimeError: If ffmpeg is not found or no devices provided.
    """
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    all_devices = input_devices + loopback_devices
    if not all_devices:
        raise RuntimeError("No devices provided for recording")

    # Filter out devices that can't be recorded from (output-only devices)
    # FFmpeg can only capture from devices with input capability
    recordable_devices = [d for d in all_devices if d.has_input]
    if not recordable_devices:
        non_recordable = [d.name for d in all_devices if not d.has_input]
        raise RuntimeError(
            f"Cannot record from selected devices (output-only): {', '.join(non_recordable)}. "
            "FFmpeg can only capture from devices with input capability. "
            "Use a loopback device (like BlackHole) for system audio capture."
        )
    if len(recordable_devices) < len(all_devices):
        skipped = [d.name for d in all_devices if not d.has_input]
        logger.warning(
            "Skipping output-only devices (cannot record from): %s", ", ".join(skipped)
        )

    all_devices = recordable_devices

    # Log device details for debugging
    logger.debug("Recording from devices:")
    for i, device in enumerate(all_devices):
        device_type = "input" if device in input_devices else "loopback"
        logger.debug("  [%d] %s (%s, id=%d)", i, device.name, device_type, device.id)

    # Build ffmpeg command
    cmd = [str(ffmpeg), "-y"]  # -y to overwrite output file

    # Add input sources (order matters: inputs first, then loopbacks)
    for device in all_devices:
        cmd.extend(["-f", "avfoundation", "-i", f":{device.id}"])
        logger.debug("Added input source: :%d (%s)", device.id, device.name)

    # Build filter complex for mixing
    if len(all_devices) > 1:
        # Multiple devices: use amix filter
        # Reference inputs as [0:a], [1:a], etc. and concatenate them
        input_labels = "".join([f"[{i}:a]" for i in range(len(all_devices))])
        num_inputs = len(all_devices)
        filter_complex = (
            f"{input_labels}amix=inputs={num_inputs}:duration=longest[a]"
        )
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[a]"])
    # Single device: no mixing needed

    # Output format
    cmd.extend(["-ar", str(sample_rate)])
    if mono:
        cmd.extend(["-ac", "1"])

    # Duration limit
    if duration:
        cmd.extend(["-t", str(duration)])

    # Output file
    cmd.append(str(output_path))

    logger.debug("recording audio: %s", " ".join(cmd))

    # Start ffmpeg process
    # If verbose, stream stderr to terminal in real-time
    if verbose:
        # Stream stderr directly to terminal for real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,  # Stream directly to terminal
        )
    else:
        # Capture stderr for error reporting
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Unbuffered
        )

    return process
