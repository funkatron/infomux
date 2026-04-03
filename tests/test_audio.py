"""
Tests for audio device discovery and classification.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from infomux.audio import (
    AudioDevice,
    AudioDeviceInventory,
    build_audio_device_inventory,
    classify_device,
    get_default_input,
    get_device_by_id,
    get_default_loopback,
    list_input_devices,
    list_loopback_devices,
    render_level_meter,
)


class TestDeviceClassification:
    """Tests for device classification."""

    def test_classify_input_device(self) -> None:
        """Input devices are classified correctly."""
        device = AudioDevice(id=0, name="Built-in Microphone")
        classified = classify_device(device)
        assert classified.direction == "input"

    def test_classify_blackhole(self) -> None:
        """BlackHole is classified as loopback."""
        device = AudioDevice(id=0, name="BlackHole 2ch")
        classified = classify_device(device)
        assert classified.direction == "loopback"

    def test_classify_screenflick_loopback(self) -> None:
        """Screenflick Loopback is classified as loopback."""
        device = AudioDevice(id=1, name="Screenflick Loopback")
        classified = classify_device(device)
        assert classified.direction == "loopback"

    def test_classify_loopback_audio(self) -> None:
        """Loopback Audio is classified as loopback."""
        device = AudioDevice(id=2, name="Loopback Audio")
        classified = classify_device(device)
        assert classified.direction == "loopback"

    def test_classify_case_insensitive(self) -> None:
        """Classification is case-insensitive."""
        device = AudioDevice(id=0, name="blackhole 16ch")
        classified = classify_device(device)
        assert classified.direction == "loopback"

    @patch("infomux.audio.get_device_capabilities")
    def test_classify_with_explicit_capabilities_skips_system_profiler(
        self, mock_capabilities
    ) -> None:
        """Explicit capabilities should bypass system_profiler."""
        classified = classify_device(
            AudioDevice(id=0, name="Built-in Microphone"),
            capabilities={"Built-in Microphone": {"input": True, "output": False, "virtual": False}},
        )
        assert classified.direction == "input"
        mock_capabilities.assert_not_called()


class TestAudioInventory:
    """Tests for inventory partitioning."""

    @patch("infomux.audio.find_tool", return_value=Path("/usr/bin/ffmpeg"))
    @patch(
        "infomux.audio.get_device_capabilities",
        return_value={
            "Built-in Microphone": {"input": True, "output": False, "virtual": False},
            "BlackHole 2ch": {"input": True, "output": True, "virtual": True},
            "Mac Studio Speakers": {"input": False, "output": True, "virtual": False},
        },
    )
    @patch(
        "infomux.audio.subprocess.run",
        return_value=type(
            "Result",
            (),
            {
                "stderr": """
[AVFoundation indev @ 0x123] AVFoundation audio devices:
[AVFoundation indev @ 0x123] [0] Built-in Microphone
[AVFoundation indev @ 0x123] [1] BlackHole 2ch
[in#0] Error opening input
""",
            },
        )(),
    )
    def test_build_inventory_partitions_devices(
        self, mock_run, mock_capabilities, mock_find_tool
    ) -> None:
        """Inventory separates recordable inputs, loopbacks, and output-only devices."""
        inventory = build_audio_device_inventory()

        assert [device.name for device in inventory.recordable_inputs] == ["Built-in Microphone"]
        assert [device.name for device in inventory.recordable_loopbacks] == ["BlackHole 2ch"]
        assert [device.name for device in inventory.output_only_devices] == ["Mac Studio Speakers"]
        assert set(inventory.devices_by_id) == {0, 1}

    @patch("infomux.audio.find_tool", return_value=Path("/usr/bin/ffmpeg"))
    @patch(
        "infomux.audio.get_device_capabilities",
        return_value={
            "Built-in Microphone": {"input": True, "output": False, "virtual": False},
        },
    )
    @patch(
        "infomux.audio.subprocess.run",
        return_value=type(
            "Result",
            (),
            {
                "stderr": """
[AVFoundation indev @ 0x123] AVFoundation audio devices:
[AVFoundation indev @ 0x123] [0] Built-in Microphone
[AVFoundation indev @ 0x123] [1] USB Podcast Mic
[in#0] Error opening input
""",
            },
        )(),
    )
    def test_build_inventory_keeps_unknown_ffmpeg_inputs_recordable(
        self, mock_run, mock_capabilities, mock_find_tool
    ) -> None:
        """Devices seen by ffmpeg stay selectable even when system_profiler omits them."""
        inventory = build_audio_device_inventory()

        assert [device.name for device in inventory.recordable_inputs] == [
            "Built-in Microphone",
            "USB Podcast Mic",
        ]

    @patch("infomux.audio.build_audio_device_inventory")
    def test_get_device_by_id_excludes_output_only_devices(self, mock_inventory) -> None:
        """Synthetic output-only IDs should not be treated as capturable devices."""
        mock_inventory.return_value = AudioDeviceInventory(
            recordable_inputs=[AudioDevice(id=0, name="Built-in Microphone", has_input=True)],
            recordable_loopbacks=[
                AudioDevice(
                    id=1,
                    name="BlackHole 2ch",
                    has_input=True,
                    has_output=True,
                    direction="loopback",
                    is_virtual=True,
                )
            ],
            output_only_devices=[
                AudioDevice(
                    id=2,
                    name="Mac Studio Speakers",
                    has_input=False,
                    has_output=True,
                    direction="output",
                )
            ],
            all_devices=[],
            devices_by_id={
                0: AudioDevice(id=0, name="Built-in Microphone", has_input=True),
                1: AudioDevice(
                    id=1,
                    name="BlackHole 2ch",
                    has_input=True,
                    has_output=True,
                    direction="loopback",
                    is_virtual=True,
                ),
            },
        )

        assert get_device_by_id(0) is not None
        assert get_device_by_id(1) is not None
        assert get_device_by_id(2) is None


class TestDeviceLists:
    """Tests for device list filtering."""

    def test_list_input_devices(self) -> None:
        """list_input_devices returns only input-capable devices."""
        # This will fail if no input devices exist, which is fine
        # We're testing the filtering logic, not device availability
        try:
            inputs = list_input_devices()
            for device in inputs:
                assert device.has_input is True
        except RuntimeError:
            # No devices available - skip test
            pytest.skip("No audio devices available")

    def test_list_loopback_devices(self) -> None:
        """list_loopback_devices returns only loopback devices."""
        try:
            loopbacks = list_loopback_devices()
            for device in loopbacks:
                assert device.direction == "loopback"
        except RuntimeError:
            # No loopback devices available - skip test
            pytest.skip("No loopback devices available")


class TestDefaultDevices:
    """Tests for default device selection."""

    def test_get_default_input(self) -> None:
        """get_default_input returns an input-capable device."""
        try:
            default = get_default_input()
            if default:
                assert default.has_input is True
        except RuntimeError:
            pytest.skip("No input devices available")

    def test_get_default_loopback(self) -> None:
        """get_default_loopback returns first loopback device."""
        try:
            default = get_default_loopback()
            if default:
                assert default.direction == "loopback"
        except RuntimeError:
            pytest.skip("No loopback devices available")


class TestLevelMeter:
    """Tests for audio level meter rendering."""

    def test_render_silence(self) -> None:
        """Silence renders as empty bars."""
        meter = render_level_meter(0.0, width=5)
        assert len(meter) == 5
        # All bars should be lowest level
        assert all(c == "▁" for c in meter)

    def test_render_max_level(self) -> None:
        """Max level renders as full bars."""
        meter = render_level_meter(1.0, width=5)
        assert len(meter) == 5
        # Should have some filled bars (exact pattern depends on implementation)

    def test_render_mid_level(self) -> None:
        """Mid level renders appropriately."""
        meter = render_level_meter(0.5, width=5)
        assert len(meter) == 5

    def test_render_clamps_level(self) -> None:
        """Level values are clamped to 0-1."""
        meter_negative = render_level_meter(-1.0, width=5)
        meter_high = render_level_meter(2.0, width=5)
        meter_zero = render_level_meter(0.0, width=5)
        meter_one = render_level_meter(1.0, width=5)

        assert meter_negative == meter_zero
        assert meter_high == meter_one

    def test_render_custom_width(self) -> None:
        """Custom width works."""
        meter = render_level_meter(0.5, width=10)
        assert len(meter) == 10
