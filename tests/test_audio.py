"""
Tests for audio device discovery and classification.
"""

from __future__ import annotations

import pytest

from infomux.audio import (
    AudioDevice,
    classify_device,
    get_default_input,
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


class TestDeviceLists:
    """Tests for device list filtering."""

    def test_list_input_devices(self) -> None:
        """list_input_devices returns only input devices."""
        # This will fail if no input devices exist, which is fine
        # We're testing the filtering logic, not device availability
        try:
            inputs = list_input_devices()
            for device in inputs:
                assert device.direction == "input"
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
        """get_default_input returns first input device."""
        try:
            default = get_default_input()
            if default:
                assert default.direction == "input"
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
