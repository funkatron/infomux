"""
Tests for the stream command and audio module.

These tests mock external dependencies (ffmpeg, whisper-stream) to test
the logic without requiring actual audio devices or transcription.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.audio import AudioDevice, list_audio_devices, select_audio_device
from infomux.commands.stream import (
    StreamMonitor,
    _find_saved_audio,
    _print_instructions,
    configure_parser,
)


class TestAudioDevice:
    """Tests for AudioDevice dataclass."""

    def test_create_audio_device(self):
        device = AudioDevice(id=0, name="Test Mic")
        assert device.id == 0
        assert device.name == "Test Mic"
        assert device.device_type == "audio"

    def test_device_with_type(self):
        device = AudioDevice(id=1, name="Camera", device_type="video")
        assert device.device_type == "video"


class TestListAudioDevices:
    """Tests for list_audio_devices function."""

    @patch("infomux.audio.find_tool")
    @patch("infomux.audio.subprocess.run")
    def test_parses_ffmpeg_output(self, mock_run, mock_find_tool):
        mock_find_tool.return_value = Path("/usr/bin/ffmpeg")
        mock_run.return_value = MagicMock(
            stderr="""
[AVFoundation indev @ 0x123] AVFoundation video devices:
[AVFoundation indev @ 0x123] [0] FaceTime Camera
[AVFoundation indev @ 0x123] AVFoundation audio devices:
[AVFoundation indev @ 0x123] [0] Built-in Microphone
[AVFoundation indev @ 0x123] [1] External USB Mic
[in#0] Error opening input
"""
        )

        devices = list_audio_devices()

        assert len(devices) == 2
        assert devices[0].id == 0
        assert devices[0].name == "Built-in Microphone"
        assert devices[1].id == 1
        assert devices[1].name == "External USB Mic"

    @patch("infomux.audio.find_tool")
    def test_raises_when_ffmpeg_not_found(self, mock_find_tool):
        mock_find_tool.return_value = None

        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            list_audio_devices()

    @patch("infomux.audio.find_tool")
    @patch("infomux.audio.subprocess.run")
    def test_returns_empty_when_no_devices(self, mock_run, mock_find_tool):
        mock_find_tool.return_value = Path("/usr/bin/ffmpeg")
        mock_run.return_value = MagicMock(
            stderr="[AVFoundation indev @ 0x123] AVFoundation audio devices:\n"
        )

        devices = list_audio_devices()
        assert devices == []


class TestSelectAudioDevice:
    """Tests for select_audio_device function."""

    def test_returns_none_when_no_devices(self, capsys):
        result = select_audio_device([])
        assert result is None
        captured = capsys.readouterr()
        assert "No audio devices found" in captured.err

    @patch("builtins.input", return_value="")
    def test_selects_default_device(self, mock_input):
        devices = [
            AudioDevice(id=0, name="Mic 1"),
            AudioDevice(id=1, name="Mic 2"),
        ]
        result = select_audio_device(devices)
        assert result == devices[0]

    @patch("builtins.input", return_value="1")
    def test_selects_specified_device(self, mock_input):
        devices = [
            AudioDevice(id=0, name="Mic 1"),
            AudioDevice(id=1, name="Mic 2"),
        ]
        result = select_audio_device(devices)
        assert result == devices[1]

    @patch("builtins.input", return_value="invalid")
    def test_returns_none_on_invalid_input(self, mock_input, capsys):
        devices = [AudioDevice(id=0, name="Mic 1")]
        result = select_audio_device(devices)
        assert result is None
        captured = capsys.readouterr()
        assert "Invalid selection" in captured.err


class TestStreamMonitor:
    """Tests for StreamMonitor class."""

    def test_check_duration_not_exceeded(self):
        mock_process = MagicMock()
        monitor = StreamMonitor(mock_process, duration=60)

        assert monitor.check_duration() is False
        assert monitor.should_stop is False

    def test_check_silence_not_exceeded(self):
        mock_process = MagicMock()
        monitor = StreamMonitor(mock_process, silence=10)

        assert monitor.check_silence() is False
        assert monitor.should_stop is False

    def test_check_stop_word_detected(self):
        mock_process = MagicMock()
        monitor = StreamMonitor(mock_process, stop_word="stop recording")

        assert monitor.check_stop_word("please stop recording now") is True
        assert monitor.should_stop is True
        assert "stop recording" in monitor.stop_reason

    def test_check_stop_word_case_insensitive(self):
        mock_process = MagicMock()
        monitor = StreamMonitor(mock_process, stop_word="STOP")

        assert monitor.check_stop_word("Please Stop Now") is True
        assert monitor.should_stop is True

    def test_check_stop_word_not_found(self):
        mock_process = MagicMock()
        monitor = StreamMonitor(mock_process, stop_word="stop")

        assert monitor.check_stop_word("continue talking") is False
        assert monitor.should_stop is False

    def test_on_speech_updates_last_speech_time(self):
        mock_process = MagicMock()
        monitor = StreamMonitor(mock_process, silence=10)
        initial_time = monitor.last_speech_time

        import time
        time.sleep(0.01)
        monitor.on_speech("hello")

        assert monitor.last_speech_time > initial_time

    def test_elapsed_time(self):
        mock_process = MagicMock()
        monitor = StreamMonitor(mock_process)

        import time
        time.sleep(0.05)

        assert monitor.elapsed() >= 0.05

    def test_remaining_time_with_duration(self):
        mock_process = MagicMock()
        monitor = StreamMonitor(mock_process, duration=60)

        remaining = monitor.remaining()
        assert remaining is not None
        assert remaining <= 60

    def test_remaining_time_without_duration(self):
        mock_process = MagicMock()
        monitor = StreamMonitor(mock_process)

        assert monitor.remaining() is None


class TestFindSavedAudio:
    """Tests for _find_saved_audio function."""

    def test_finds_wav_in_cwd(self, tmp_path, monkeypatch):
        # Create a wav file in cwd
        monkeypatch.chdir(tmp_path)
        wav_file = tmp_path / "20260101120000.wav"
        wav_file.write_bytes(b"fake audio")

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        result = _find_saved_audio(run_dir)

        assert result is not None
        assert result.name == "audio_full.wav"
        assert result.parent == run_dir

    def test_finds_audio_in_run_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        audio_file = run_dir / "audio_full.wav"
        audio_file.write_bytes(b"fake audio")

        result = _find_saved_audio(run_dir)

        assert result is not None
        assert result == audio_file

    def test_returns_none_when_no_audio(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        result = _find_saved_audio(run_dir)
        assert result is None


class TestConfigureParser:
    """Tests for configure_parser function."""

    def test_parser_has_all_options(self):
        import argparse
        parser = argparse.ArgumentParser()
        configure_parser(parser)

        # Parse with minimal args
        args = parser.parse_args([])

        assert hasattr(args, "device")
        assert hasattr(args, "list_devices")
        assert hasattr(args, "no_save_audio")
        assert hasattr(args, "language")
        assert hasattr(args, "duration")
        assert hasattr(args, "silence")
        assert hasattr(args, "stop_word")

    def test_default_stop_word(self):
        import argparse
        parser = argparse.ArgumentParser()
        configure_parser(parser)

        args = parser.parse_args([])
        assert args.stop_word == "stop recording"

    def test_default_language(self):
        import argparse
        parser = argparse.ArgumentParser()
        configure_parser(parser)

        args = parser.parse_args([])
        assert args.language == "en"


class TestPrintInstructions:
    """Tests for _print_instructions function."""

    def test_prints_device_name(self, capsys):
        device = AudioDevice(id=0, name="Test Microphone")
        _print_instructions(device, None, None, None)

        captured = capsys.readouterr()
        assert "Test Microphone" in captured.err
        assert "Ctrl+C" in captured.err

    def test_prints_duration_when_set(self, capsys):
        device = AudioDevice(id=0, name="Mic")
        _print_instructions(device, duration=60, silence=None, stop_word=None)

        captured = capsys.readouterr()
        assert "60 seconds" in captured.err
        assert "auto-stop" in captured.err

    def test_prints_silence_when_set(self, capsys):
        device = AudioDevice(id=0, name="Mic")
        _print_instructions(device, duration=None, silence=10, stop_word=None)

        captured = capsys.readouterr()
        assert "10 seconds" in captured.err
        assert "silent" in captured.err

    def test_prints_stop_word_when_set(self, capsys):
        device = AudioDevice(id=0, name="Mic")
        _print_instructions(device, None, None, stop_word="end session")

        captured = capsys.readouterr()
        assert "end session" in captured.err
