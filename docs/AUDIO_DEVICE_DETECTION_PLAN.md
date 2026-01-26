# Audio Device Detection Enhancement Plan

## Overview

Enhance the audio device detection system to properly identify devices with both input and output capabilities, allowing devices like M2 and OWC Thunderbolt to appear in both input and output lists.

## Requirements

- Devices with both input and output capabilities should appear in both INPUT and OUTPUT lists
- Output-only devices (like Mac Studio Speakers) should appear in OUTPUT list
- Default device selection should prefer:
  - Non-virtual devices for input (e.g., M2 microphone)
  - Loopback devices for output (e.g., BlackHole for system audio capture)
- Safety checks to prevent recording from output-only devices

## Implementation

### Device Capability Detection

**Approach:** Use `system_profiler SPAudioDataType` to query device capabilities

- Parse `system_profiler` output to detect:
  - Input channels → `has_input = True`
  - Output channels → `has_output = True`
  - Transport: Virtual → `is_virtual = True`

**Implementation:**
- `get_device_capabilities()` function queries system_profiler
- Returns dictionary mapping device name → capabilities dict

### Enhanced AudioDevice Model

**Changes to `AudioDevice` dataclass:**
- Added `has_input: bool` field (default: True)
- Added `has_output: bool` field (default: False)
- Added `is_virtual: bool` field (default: False)
- Updated `direction` field to support: "input", "output", "loopback", "both"

**Device Classification:**
- `classify_device()` now uses `get_device_capabilities()` to set capabilities
- Virtual devices are marked as loopback
- Devices with both input and output get `direction = "both"`

### Device Lists

**Input Devices (`list_input_devices()`):**
- Returns all devices with `has_input = True`
- Includes: M2, OWC Thunderbolt, BlackHole, Screenflick Loopback

**Output Devices (`list_output_devices()`):**
- Returns all devices with `has_output = True` OR `is_virtual = True`
- Includes devices from ffmpeg list PLUS output-only devices from system_profiler
- Includes: M2, OWC Thunderbolt, BlackHole, Screenflick, Mac Studio Speakers, DELL monitors

**Loopback Devices (`list_loopback_devices()`):**
- Returns devices with `direction = "loopback"` OR `is_virtual = True`
- Used for system audio capture

### Default Device Selection

**Default Input (`get_default_input()`):**
- Prefers non-virtual devices (actual microphones)
- Falls back to virtual devices if no physical input available
- Example: M2 (not BlackHole)

**Default Output (`get_default_output()`):**
- Prefers loopback devices (for system audio capture)
- Falls back to any output-capable device
- Example: BlackHole 2ch

### Safety Checks

**Recording Validation:**
- `record_audio()` filters out devices without `has_input = True`
- Raises clear error if only output-only devices are selected
- Warns if output-only devices are skipped

### User Interface

**`--list-devices` Output:**
- Shows INPUTS section with capability markers: `(input, output, virtual)`
- Shows OUTPUTS section with markers: `[loopback]`, `[output-only]`
- Clear indication of which devices can be recorded from

**Interactive Selection (`--prompt`):**
- Shows devices in both INPUT and OUTPUT sections when applicable
- Live audio meters for all devices
- Clear markers for device capabilities

## Testing

### Device Detection Tests
- Verify M2 appears in both input and output lists
- Verify OWC Thunderbolt appears in both input and output lists
- Verify Mac Studio Speakers appears only in output list (marked as output-only)
- Verify BlackHole appears in both lists (marked as loopback)

### Default Selection Tests
- Verify default input prefers M2 over BlackHole
- Verify default output prefers BlackHole over M2

### Safety Tests
- Verify recording fails gracefully if only output-only devices selected
- Verify warning shown when output-only devices are skipped

## Files Modified

- `src/infomux/audio.py`
  - Added `get_device_capabilities()` function
  - Enhanced `AudioDevice` dataclass
  - Updated `classify_device()` function
  - Updated `list_input_devices()`, `list_output_devices()`, `list_loopback_devices()`
  - Updated `get_default_input()`, `get_default_output()`
  - Added safety check in `record_audio()`

- `src/infomux/commands/stream.py`
  - Updated `_list_devices()` to show capabilities
  - Updated `interactive_device_selection()` to use output devices
  - Updated `_display_devices_with_meters()` to show capabilities

- `tests/test_audio.py`
  - Added tests for device classification
  - Added tests for device lists
  - Added tests for default selection

## Status

✅ **Completed** - Feature implemented and tested

## Future Enhancements

- Support for selecting multiple input/output devices simultaneously
- Visual indicators for active/default devices
- Device capability caching to avoid repeated system_profiler calls
- Support for device aliases/nicknames
- Better error messages for missing loopback devices
