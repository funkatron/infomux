"""
Functional tests for lyric video pipelines.

Tests pipeline execution with vocal isolation and forced alignment:
- lyric-video-vocals: isolate_vocals → transcribe_timed → extract_audio → generate_lyric_video
- lyric-video-aligned: isolate_vocals → align_lyrics → extract_audio → generate_lyric_video

Test example: "These Days" by Dogtablet
Note: Full execution tests are skipped by default as they require:
- Real audio files (recommend using first 60 seconds for faster testing: ffmpeg -i input.mp3 -t 60 -c copy sample.mp3)
- External tools (ffmpeg, demucs/spleeter, whisper-cli, aeneas)
- Official lyrics file for aligned pipelines
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.job import JobEnvelope, JobStatus
from infomux.pipeline import run_pipeline
from infomux.pipeline_def import get_pipeline


class TestLyricVideoVocalsPipeline:
    """Functional tests for lyric-video-vocals pipeline."""

    def test_pipeline_exists(self) -> None:
        """Pipeline is registered and accessible."""
        pipeline = get_pipeline("lyric-video-vocals")
        assert pipeline.name == "lyric-video-vocals"
        assert len(pipeline.steps) == 4

    def test_pipeline_steps_order(self) -> None:
        """Pipeline steps are in correct order."""
        pipeline = get_pipeline("lyric-video-vocals")
        step_names = pipeline.step_names()
        assert step_names == [
            "isolate_vocals",
            "transcribe_timed",
            "extract_audio",
            "generate_lyric_video",
        ]

    def test_pipeline_step_dependencies(self) -> None:
        """Pipeline steps have correct input dependencies."""
        pipeline = get_pipeline("lyric-video-vocals")

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

    def test_pipeline_execution_with_real_file(self, tmp_path: Path, monkeypatch) -> None:
        """
        Test pipeline execution with real audio file.
        
        This is a functional test that requires:
        - Real audio file at the specified path
        - ffmpeg, demucs, whisper-cli installed
        - Sufficient time to run (may take several minutes)
        
        Marked as skip if file doesn't exist to avoid failures in CI.
        """
        # Use test fixture audio file (60-second sample)
        audio_file = Path(__file__).parent / "fixtures" / "test-audio-sample.mp3"
        
        if not audio_file.exists():
            pytest.skip(f"Test audio file not found: {audio_file}")

        # Set data directory to temp path for test isolation
        # But keep the whisper model path pointing to the real model location
        monkeypatch.setenv("INFOMUX_DATA_DIR", str(tmp_path))
        import os
        if "INFOMUX_WHISPER_MODEL" not in os.environ:
            # Set the whisper model path explicitly so it's found even with custom DATA_DIR
            model_path = Path.home() / ".local" / "share" / "infomux" / "models" / "whisper" / "ggml-base.en.bin"
            if model_path.exists():
                monkeypatch.setenv("INFOMUX_WHISPER_MODEL", str(model_path))
        
        # Create job
        from infomux.job import InputFile
        input_file = InputFile.from_path(audio_file)
        job = JobEnvelope.create(input_file=input_file)
        
        # Get pipeline
        pipeline = get_pipeline("lyric-video-vocals")
        
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
        
        # Verify pipeline completed successfully
        assert success, f"Pipeline failed. Check job: {job.id}"
        
        # Verify all steps completed
        assert len(job.steps) == 4, f"Expected 4 steps, got {len(job.steps)}"
        for step in job.steps:
            assert step.status == "completed", f"Step {step.name} failed: {step.error}"
        
        # Verify outputs exist
        assert (run_dir / "audio_vocals_only.wav").exists(), "isolate_vocals should create audio_vocals_only.wav"
        assert (run_dir / "transcript.json").exists(), "transcribe_timed should create transcript.json"
        assert (run_dir / "audio_full.wav").exists(), "extract_audio should create audio_full.wav"
        
        # Find lyric video output
        video_files = list(run_dir.glob("*_lyric_video.mp4"))
        assert len(video_files) > 0, "generate_lyric_video should create lyric video"
        
        # Verify transcript.json has word-level timestamps
        with open(run_dir / "transcript.json", encoding="utf-8", errors="replace") as f:
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
        pipeline = get_pipeline("lyric-video-vocals")
        
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


class TestLyricVideoAlignedPipeline:
    """Functional tests for lyric-video-aligned pipelines using official lyrics."""

    # Official lyrics for "These Days" by Dogtablet
    # Used for testing aligned pipelines with forced alignment
    TEST_LYRICS = """When you walk through the door .
Hearts hit the floor.
The sky fades to north and south
Your lips, those words, these days, your mouth. 

Five times by design,
I'm still alive . 

I hold your shadow, and feel your heat, become the centre and all you need. 
Blaze alive , 
Shrink and hold. 
Hearts stop. Worlds end. Prime control.

Five times by design. I'm still alive . 

When you walk through the door. Hearts hit the floor. The sky fades from north to south…
Your lips, those words, these days, your mouth. 

Five times by design. I'm still alive."""

    def test_aligned_pipeline_exists(self) -> None:
        """Aligned pipeline is registered and accessible."""
        pipeline = get_pipeline("lyric-video-aligned")
        assert pipeline.name == "lyric-video-aligned"
        assert len(pipeline.steps) == 4  # Includes isolate_vocals

    def test_aligned_pipeline_has_isolation(self) -> None:
        """Aligned pipeline includes vocal isolation."""
        pipeline = get_pipeline("lyric-video-aligned")
        assert "isolate_vocals" in pipeline.step_names()
        assert len(pipeline.steps) == 4

    def test_aligned_pipeline_steps_order(self) -> None:
        """Aligned pipeline steps are in correct order."""
        pipeline = get_pipeline("lyric-video-aligned")
        step_names = pipeline.step_names()
        assert step_names == [
            "isolate_vocals",
            "align_lyrics",
            "extract_audio",
            "generate_lyric_video",
        ]


    def test_aligned_pipeline_step_dependencies(self) -> None:
        """Aligned pipeline steps have correct input dependencies."""
        pipeline = get_pipeline("lyric-video-aligned")

        isolate_vocals = pipeline.get_step("isolate_vocals")
        assert isolate_vocals is not None
        assert isolate_vocals.input_from is None  # Uses original input

        align_lyrics = pipeline.get_step("align_lyrics")
        assert align_lyrics is not None
        assert align_lyrics.input_from == "isolate_vocals"  # Aligns to isolated vocals (better accuracy)

        extract_audio = pipeline.get_step("extract_audio")
        assert extract_audio is not None
        assert extract_audio.input_from is None  # Uses original input

        generate_lyric_video = pipeline.get_step("generate_lyric_video")
        assert generate_lyric_video is not None
        assert generate_lyric_video.input_from == "extract_audio"  # Uses extracted audio (with music)

    def test_lyrics_file_reading(self, tmp_path: Path) -> None:
        """Test that lyrics file can be read and used by align_lyrics step."""
        from infomux.steps.align_lyrics import AlignLyricsStep
        from infomux.steps import StepError

        # Create a test lyrics file
        lyrics_file = tmp_path / "lyrics.txt"
        lyrics_file.write_text(self.TEST_LYRICS, encoding="utf-8")

        # Create a dummy audio file
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"fake audio")

        # Test that step can find and read lyrics file when provided
        step = AlignLyricsStep(lyrics_file=str(lyrics_file))
        
        # The step should be able to find the lyrics file
        # We can't fully execute without aeneas, but we can test the file reading logic
        # by checking that it raises an appropriate error when aeneas is not available
        # (rather than a "file not found" error)
        try:
            step.execute(audio_file, tmp_path)
        except StepError as e:
            # Should fail on aeneas not found, not on lyrics file not found
            assert "lyrics file not found" not in str(e).lower(), \
                f"Should have found lyrics file, but got: {e}"
            # It's OK if it fails on aeneas - that means it found the lyrics file

        # Test that step can find lyrics.txt in output_dir automatically
        step2 = AlignLyricsStep(lyrics_file=None)  # No explicit file
        try:
            step2.execute(audio_file, tmp_path)
        except StepError as e:
            # Should fail on aeneas not found, not on lyrics file not found
            assert "lyrics file not found" not in str(e).lower(), \
                f"Should have auto-found lyrics.txt, but got: {e}"

        # Test that step raises error when lyrics file doesn't exist
        step3 = AlignLyricsStep(lyrics_file="nonexistent.txt")
        with pytest.raises(StepError) as exc_info:
            step3.execute(audio_file, tmp_path)
        assert "lyrics file not found" in str(exc_info.value).lower()

        # Test that step raises error when lyrics file is empty
        empty_lyrics = tmp_path / "empty_lyrics.txt"
        empty_lyrics.write_text("", encoding="utf-8")
        step4 = AlignLyricsStep(lyrics_file=str(empty_lyrics))
        with pytest.raises(StepError) as exc_info:
            step4.execute(audio_file, tmp_path)
        assert "empty" in str(exc_info.value).lower()

    def test_aligned_pipeline_with_lyrics_file(self, tmp_path: Path, monkeypatch) -> None:
        """
        Test aligned pipeline execution with official lyrics file.
        
        This is a functional test that requires:
        - Real audio file (recommend using first 60 seconds: ffmpeg -i input.mp3 -t 60 -c copy sample.mp3)
        - Lyrics file with official lyrics matching the audio
        - ffmpeg, aeneas, demucs installed
        - Sufficient time to run (may take several minutes for full song)
        
        This test demonstrates using official lyrics for more accurate timing than transcription.
        Example: "These Days" by Dogtablet
        """
        # Check if aeneas is available
        import sys
        try:
            result = subprocess.run(
                [sys.executable, "-m", "aeneas.tools.execute_task", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("aeneas not installed - required for forced alignment")
        
        if result.returncode != 0:
            pytest.skip("aeneas not available - required for forced alignment")
        
        # Use test fixture audio file (60-second sample)
        audio_file = Path(__file__).parent / "fixtures" / "test-audio-sample.mp3"
        
        if not audio_file.exists():
            pytest.skip(f"Test audio file not found: {audio_file}")

        # Set data directory to temp path for test isolation
        # But keep the whisper model path pointing to the real model location
        monkeypatch.setenv("INFOMUX_DATA_DIR", str(tmp_path))
        import os
        if "INFOMUX_WHISPER_MODEL" not in os.environ:
            # Set the whisper model path explicitly so it's found even with custom DATA_DIR
            model_path = Path.home() / ".local" / "share" / "infomux" / "models" / "whisper" / "ggml-base.en.bin"
            if model_path.exists():
                monkeypatch.setenv("INFOMUX_WHISPER_MODEL", str(model_path))
        
        # Create job
        from infomux.job import InputFile
        input_file = InputFile.from_path(audio_file)
        job = JobEnvelope.create(input_file=input_file)
        
        # Get pipeline
        pipeline = get_pipeline("lyric-video-aligned")
        
        # Create run directory
        run_dir = tmp_path / "runs" / job.id
        run_dir.mkdir(parents=True)
        
        # Copy input file to run directory
        import shutil
        run_input = run_dir / audio_file.name
        shutil.copy2(audio_file, run_input)
        job.input.path = str(run_input)
        
        # Create lyrics file in run directory
        lyrics_file = run_dir / "lyrics.txt"
        lyrics_file.write_text(self.TEST_LYRICS, encoding="utf-8")
        
        # Update pipeline step config to use lyrics file
        align_step = pipeline.get_step("align_lyrics")
        if align_step:
            align_step.config["lyrics_file"] = str(lyrics_file)
        
        # Run pipeline
        success = run_pipeline(
            job=job,
            run_dir=run_dir,
            pipeline=pipeline,
        )
        
        # Verify pipeline completed successfully
        assert success, f"Pipeline failed. Check job: {job.id}"
        
        # Verify all steps completed
        assert len(job.steps) == 4, f"Expected 4 steps, got {len(job.steps)}"
        for step in job.steps:
            assert step.status == "completed", f"Step {step.name} failed: {step.error}"
        
        # Verify outputs exist
        assert (run_dir / "audio_vocals_only.wav").exists(), "isolate_vocals should create audio_vocals_only.wav"
        assert (run_dir / "transcript.json").exists(), "align_lyrics should create transcript.json"
        assert (run_dir / "audio_full.wav").exists(), "extract_audio should create audio_full.wav"
        
        # Find lyric video output
        video_files = list(run_dir.glob("*_lyric_video.mp4"))
        assert len(video_files) > 0, "generate_lyric_video should create lyric video"
        
        # Verify transcript.json has word-level timestamps matching official lyrics
        with open(run_dir / "transcript.json", encoding="utf-8", errors="replace") as f:
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
        
        # Verify lyrics text matches (first few words)
        transcript_text = " ".join(
            token["text"].strip() for segment in transcription
            for token in segment.get("tokens", [])
        )
        # Check that first words from lyrics appear in transcript
        first_lyrics_words = " ".join(self.TEST_LYRICS.split()[:5]).lower()
        assert first_lyrics_words.lower() in transcript_text.lower(), \
            f"Transcript should contain lyrics. First words: {first_lyrics_words}"
