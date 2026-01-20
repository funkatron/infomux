"""
Functional tests for lyric video pipelines.

Tests pipeline execution with vocal isolation and forced alignment:
- lyric-video-isolated: isolate_vocals → transcribe_timed → extract_audio → generate_lyric_video
- lyric-video-aligned: extract_audio → align_lyrics → generate_lyric_video
- lyric-video-aligned-isolated: isolate_vocals → align_lyrics → extract_audio → generate_lyric_video

Note: Full execution tests are skipped by default as they require:
- Real audio files (recommend using first 60 seconds for faster testing)
- External tools (ffmpeg, demucs/spleeter, whisper-cli, aeneas)
- Official lyrics file for aligned pipelines
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


class TestLyricVideoAlignedPipeline:
    """Functional tests for lyric-video-aligned pipelines using official lyrics."""

    # Example official lyrics for testing aligned pipelines
    # Note: Replace with your own lyrics file for actual testing
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
        assert len(pipeline.steps) == 3

    def test_aligned_isolated_pipeline_exists(self) -> None:
        """Aligned isolated pipeline is registered and accessible."""
        pipeline = get_pipeline("lyric-video-aligned-isolated")
        assert pipeline.name == "lyric-video-aligned-isolated"
        assert len(pipeline.steps) == 4

    def test_aligned_pipeline_steps_order(self) -> None:
        """Aligned pipeline steps are in correct order."""
        pipeline = get_pipeline("lyric-video-aligned")
        step_names = pipeline.step_names()
        assert step_names == [
            "extract_audio",
            "align_lyrics",
            "generate_lyric_video",
        ]

    def test_aligned_isolated_pipeline_steps_order(self) -> None:
        """Aligned isolated pipeline steps are in correct order."""
        pipeline = get_pipeline("lyric-video-aligned-isolated")
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

        extract_audio = pipeline.get_step("extract_audio")
        assert extract_audio is not None
        assert extract_audio.input_from is None  # Uses original input

        align_lyrics = pipeline.get_step("align_lyrics")
        assert align_lyrics is not None
        assert align_lyrics.input_from == "extract_audio"  # Aligns to extracted audio

        generate_lyric_video = pipeline.get_step("generate_lyric_video")
        assert generate_lyric_video is not None
        assert generate_lyric_video.input_from == "extract_audio"  # Uses extracted audio (with music)

    def test_aligned_isolated_pipeline_step_dependencies(self) -> None:
        """Aligned isolated pipeline steps have correct input dependencies."""
        pipeline = get_pipeline("lyric-video-aligned-isolated")

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

    @pytest.mark.skip(
        reason="Requires real audio file, lyrics file, and external tools (ffmpeg, aeneas, demucs). "
        "Run manually with: infomux run --pipeline lyric-video-aligned-isolated --lyrics-file lyrics.txt <audio-file>"
    )
    def test_aligned_pipeline_with_lyrics_file(self, tmp_path: Path, monkeypatch) -> None:
        """
        Test aligned pipeline execution with official lyrics file.
        
        This is a functional test that requires:
        - Real audio file (recommend using first 60 seconds: ffmpeg -i input.mp3 -t 60 -c copy sample.mp3)
        - Lyrics file with official lyrics matching the audio
        - ffmpeg, aeneas, demucs installed
        - Sufficient time to run (may take several minutes for full song)
        
        This test demonstrates using official lyrics for more accurate timing than transcription.
        """
        # Example: Use a test audio file (first 60 seconds recommended for faster testing)
        # audio_file = Path("/path/to/your/test-audio-sample.mp3")
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
        pipeline = get_pipeline("lyric-video-aligned-isolated")
        
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
        
        # Verify pipeline completed
        assert success, f"Pipeline failed. Check job: {job.id}"
        assert job.status == JobStatus.COMPLETED.value
        
        # Verify all steps completed
        assert len(job.steps) == 4
        for step in job.steps:
            assert step.status == "completed", f"Step {step.name} failed: {step.error}"
        
        # Verify outputs exist
        assert (run_dir / "audio_vocals.wav").exists(), "isolate_vocals should create audio_vocals.wav"
        assert (run_dir / "transcript.json").exists(), "align_lyrics should create transcript.json"
        assert (run_dir / "audio.wav").exists(), "extract_audio should create audio.wav"
        
        # Find lyric video output
        video_files = list(run_dir.glob("*_lyric_video.mp4"))
        assert len(video_files) > 0, "generate_lyric_video should create lyric video"
        
        # Verify transcript.json has word-level timestamps matching official lyrics
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
        
        # Verify lyrics text matches (first few words)
        transcript_text = " ".join(
            token["text"].strip() for segment in transcription
            for token in segment.get("tokens", [])
        )
        # Check that first words from lyrics appear in transcript
        first_lyrics_words = " ".join(self.TEST_LYRICS.split()[:5]).lower()
        assert first_lyrics_words.lower() in transcript_text.lower(), \
            f"Transcript should contain lyrics. First words: {first_lyrics_words}"
