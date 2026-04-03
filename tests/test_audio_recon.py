"""
Tests for audio-recon command.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from infomux.audio import AudioDevice, choose_recon_capture_device
from infomux.commands import audio_recon as ar


class TestChooseReconCaptureDevice:
    """Tests for choose_recon_capture_device heuristics."""

    @patch("infomux.audio.list_output_devices")
    def test_prefers_aggregate_over_blackhole(self, mock_list: MagicMock) -> None:
        mock_list.return_value = [
            AudioDevice(id=0, name="BlackHole 2ch", has_input=True, has_output=True),
            AudioDevice(
                id=1,
                name="infomux-aggregate-device",
                has_input=True,
                has_output=True,
            ),
        ]
        d = choose_recon_capture_device()
        assert d is not None
        assert d.name == "infomux-aggregate-device"

    @patch.dict("os.environ", {"INFOMUX_RECON_CAPTURE": "BlackHole 2ch"}, clear=False)
    @patch("infomux.audio.list_output_devices")
    def test_env_override(self, mock_list: MagicMock) -> None:
        mock_list.return_value = [
            AudioDevice(id=0, name="Other", has_input=True),
            AudioDevice(id=2, name="BlackHole 2ch", has_input=True),
        ]
        d = choose_recon_capture_device()
        assert d is not None
        assert d.name == "BlackHole 2ch"


class TestAudioReconExecute:
    """Tests for audio_recon.execute."""

    def test_check_only_exits_zero(self) -> None:
        dev = AudioDevice(id=1, name="infomux-aggregate-device", has_input=True)
        args = Namespace(
            duration=8,
            silence_threshold=-80.0,
            output_id=None,
            output_name=None,
            switch_output=None,
            sleep_after_switch=None,
            check_only=True,
            play=False,
            json=False,
            quiet=True,
        )
        with patch.object(ar, "_resolve_capture_device", return_value=dev):
            assert ar.execute(args) == ar.EXIT_PASS

    def test_json_check_only(self) -> None:
        dev = AudioDevice(id=1, name="d", has_input=True)
        args = Namespace(
            duration=8,
            silence_threshold=-80.0,
            output_id=None,
            output_name=None,
            switch_output=None,
            sleep_after_switch=None,
            check_only=True,
            play=False,
            json=True,
            quiet=True,
        )
        with patch.object(ar, "_resolve_capture_device", return_value=dev):
            with patch("builtins.print") as mock_print:
                code = ar.execute(args)
        assert code == ar.EXIT_PASS
        assert mock_print.called

    def test_pass_when_levels_not_silent(self, tmp_path: Path) -> None:
        dev = AudioDevice(id=1, name="d", has_input=True)

        args = Namespace(
            duration=2,
            silence_threshold=-80.0,
            output_id=None,
            output_name=None,
            switch_output=None,
            sleep_after_switch=None,
            check_only=False,
            play=False,
            json=False,
            quiet=True,
        )

        def record_side_effect(**kwargs: object) -> MagicMock:
            op = kwargs["output_path"]
            assert isinstance(op, Path)
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_bytes(b"RIFFfake")
            proc = MagicMock()
            proc.wait = MagicMock(return_value=0)
            return proc

        with patch.object(ar, "_resolve_capture_device", return_value=dev):
            with patch("infomux.commands.audio_recon.record_audio", side_effect=record_side_effect):
                with patch("infomux.commands.audio_recon.get_run_dir", return_value=tmp_path):
                    with patch("infomux.commands.audio_recon.JobEnvelope.create") as jc:
                        jc.return_value.id = "run-test"
                        with patch.object(ar, "_wav_peak_db", return_value=(-20.0, -30.0)):
                            code = ar.execute(args)

        assert code == ar.EXIT_PASS

    def test_silent_exit_code(self, tmp_path: Path) -> None:
        dev = AudioDevice(id=1, name="d", has_input=True)
        args = Namespace(
            duration=2,
            silence_threshold=-80.0,
            output_id=None,
            output_name=None,
            switch_output=None,
            sleep_after_switch=None,
            check_only=False,
            play=False,
            json=False,
            quiet=True,
        )

        def record_side_effect(**kwargs: object) -> MagicMock:
            op = kwargs["output_path"]
            assert isinstance(op, Path)
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_bytes(b"RIFFfake")
            proc = MagicMock()
            proc.wait = MagicMock(return_value=0)
            return proc

        with patch.object(ar, "_resolve_capture_device", return_value=dev):
            with patch("infomux.commands.audio_recon.record_audio", side_effect=record_side_effect):
                with patch("infomux.commands.audio_recon.get_run_dir", return_value=tmp_path):
                    with patch("infomux.commands.audio_recon.JobEnvelope.create") as jc:
                        jc.return_value.id = "run-test"
                        with patch.object(ar, "_wav_peak_db", return_value=(-91.0, -91.0)):
                            code = ar.execute(args)

        assert code == ar.EXIT_SILENT
