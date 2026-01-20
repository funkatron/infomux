"""
Functional test for lyric-video-isolated pipeline.

Tests the full pipeline execution with vocal isolation:
isolate_vocals → transcribe_timed → extract_audio → generate_lyric_video
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.job import JobEnvelope, JobStatus
from infomux.pipeline import run_pipeline
from infomux.pipeline_def import get_pipeline


class TestLyricVideoIsolatedPipeline:
    """Functional tests for lyric-video-isolated pipeline."""

    def test_pipeline_exists(self) -> None:
        """Pipeline is registered and accessible."""
        pipeline = get_pipeline("lyric-video-isolated")
        assert pipeline.name == "lyric-video-isolated"
        assert len(pipeline.steps) == 4

    def test_pipeline_steps_order(self) -> None:
        """Pipeline steps are in correct order."""
        pipeline = get_pipeline("lyric-video-isolated")
        step_names = pipeline.step_names()
        assert step_names == [
            "isolate_vocals",
            "transcribe_timed",
            "extract_audio",
            "generate_lyric_video",
        ]

    def test_pipeline_step_dependencies(self) -> None:
        """Pipeline steps have correct input dependencies."""
        pipeline = get_pipeline("lyric-video-isolated")

        isolate_vocals = pipeline.get_step("isolate_vocals")
        assert isolate_vocals is not None
        assert isolate_vocals.input_from is None  # Uses original input

        transcribe_timed = pipeline.get_step("transcribe_timed")
        assert transcribe_timed is not None
        assert transcribe_timed.input_from == "isolate_vocals"  # Uses isolated vocals

        extract_audio = pipeline.get_step("extract_audio")
        assert extract_audio is not None
        assert extract_audio.input_from is None  # Uses original input

        generate_lyric_video = pipeline.get_step("generate_lyric_video")
        assert generate_lyric_video is not None
        assert generate_lyric_video.input_from == "extract_audio"  # Uses extracted audio (with music)

    @pytest.mark.skip(
        reason="Requires real audio file and external tools (ffmpeg, demucs, whisper-cli). "
        "Run manually with: infomux run --pipeline lyric-video-isolated <audio-file>"
    )
    def test_pipeline_execution_with_real_file(self, tmp_path: Path, monkeypatch) -> None:
        """
        Test pipeline execution with real audio file.
        
        This is a functional test that requires:
        - Real audio file at the specified path
        - ffmpeg, demucs, whisper-cli installed
        - Sufficient time to run (may take several minutes)
        
        Marked as skip if file doesn't exist to avoid failures in CI.
        """
        # Use the actual file path from the user's example
        audio_file = Path(
            "/Users/coj/Library/Mobile Documents/com~apple~CloudDocs/_TRANSFER/"
            "%5BTEST%20MASTER%20002%5D%20-%20Dogtablet%20-%20These%20Days%20-%202Bit%20Through%20The%20Wormhole%20Edit%20-%20EAM%20Mix-05.mp3"
        )
        
        if not audio_file.exists():
            pytest.skip(f"Test audio file not found: {audio_file}")

        # Set data directory to temp path for test isolation
        monkeypatch.setenv("INFOMUX_DATA_DIR", str(tmp_path))
        
        # Create job
        from infomux.job import InputFile
        input_file = InputFile.from_path(audio_file)
        job = JobEnvelope.create(input_file=input_file)
        
        # Get pipeline
        pipeline = get_pipeline("lyric-video-isolated")
        
        # Create run directory
        run_dir = tmp_path / "runs" / job.id
        run_dir.mkdir(parents=True)
        
        # Copy input file to run directory (simulate what the pipeline does)
        import shutil
        run_input = run_dir / audio_file.name
        shutil.copy2(audio_file, run_input)
        job.input.path = str(run_input)
        
        # Run pipeline
        success = run_pipeline(
            job=job,
            run_dir=run_dir,
            pipeline=pipeline,
        )
        
        # Verify pipeline completed
        assert success, f"Pipeline failed. Check job: {job.id}"
        assert job.status == JobStatus.COMPLETED.value
        
        # Verify all steps completed
        assert len(job.steps) == 4
        for step in job.steps:
            assert step.status == "completed", f"Step {step.name} failed: {step.error}"
        
        # Verify outputs exist
        assert (run_dir / "audio_vocals.wav").exists(), "isolate_vocals should create audio_vocals.wav"
        assert (run_dir / "transcript.json").exists(), "transcribe_timed should create transcript.json"
        assert (run_dir / "audio.wav").exists(), "extract_audio should create audio.wav"
        
        # Find lyric video output
        video_files = list(run_dir.glob("*_lyric_video.mp4"))
        assert len(video_files) > 0, "generate_lyric_video should create lyric video"
        
        # Verify transcript.json has word-level timestamps
        with open(run_dir / "transcript.json", encoding="utf-8") as f:
            transcript_data = json.load(f)
        
        assert "transcription" in transcript_data
        transcription = transcript_data["transcription"]
        assert len(transcription) > 0, "transcript.json should have transcription data"
        
        # Verify first segment has tokens with timestamps
        first_segment = transcription[0]
        assert "tokens" in first_segment
        assert len(first_segment["tokens"]) > 0, "First segment should have tokens"
        
        first_token = first_segment["tokens"][0]
        assert "timestamps" in first_token
        assert "from" in first_token["timestamps"]
        assert "to" in first_token["timestamps"]

    def test_pipeline_config_values(self) -> None:
        """Pipeline step configs have correct values."""
        pipeline = get_pipeline("lyric-video-isolated")
        
        isolate_vocals = pipeline.get_step("isolate_vocals")
        assert isolate_vocals is not None
        assert isolate_vocals.config.get("tool") == "demucs"
        
        transcribe_timed = pipeline.get_step("transcribe_timed")
        assert transcribe_timed is not None
        assert transcribe_timed.config.get("generate_word_level") is True
        
        generate_lyric_video = pipeline.get_step("generate_lyric_video")
        assert generate_lyric_video is not None
        assert generate_lyric_video.config.get("font_size") == 48
        assert generate_lyric_video.config.get("background_color") == "black"
