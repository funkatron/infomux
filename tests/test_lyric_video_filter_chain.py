"""
Test filter chain construction for lyric video with many words.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from infomux.steps.generate_lyric_video import (
    GenerateLyricVideoStep,
    PositionedWord,
    WordEntry,
)


def test_filter_chain_with_many_words(tmp_path: Path) -> None:
    """Test that filter chain is correctly constructed with many words (like 162 words)."""
    # Create a transcript.json with many words
    json_file = tmp_path / "transcript.json"
    
    # Generate 162 words with timestamps
    transcription = []
    tokens = []
    for i in range(162):
        start_ms = i * 1000  # 1 second per word
        end_ms = start_ms + 500  # 0.5 second duration
        
        # Format timestamps as HH:MM:SS,mmm
        start_h = start_ms // 3600000
        start_m = (start_ms % 3600000) // 60000
        start_s = (start_ms % 60000) // 1000
        start_ms_remainder = start_ms % 1000
        
        end_h = end_ms // 3600000
        end_m = (end_ms % 3600000) // 60000
        end_s = (end_ms % 60000) // 1000
        end_ms_remainder = end_ms % 1000
        
        from_ts = f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms_remainder:03d}"
        to_ts = f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms_remainder:03d}"
        
        # Add word boundary token
        tokens.append({
            "text": f" Word{i}",
            "timestamps": {"from": from_ts, "to": to_ts},
        })
    
    transcription.append({"tokens": tokens})
    json_data = {"transcription": transcription}
    json_file.write_text(json.dumps(json_data), encoding="utf-8")
    
    # Create audio file
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake audio")
    
    step = GenerateLyricVideoStep()
    
    # Parse words
    words = step._parse_word_timestamps(json_file)
    assert len(words) == 162
    
    # Calculate positions
    positioned = step._calculate_positions(words, 1920, 1080)
    assert len(positioned) == 162
    
    # Build command
    with patch("infomux.steps.generate_lyric_video.get_tool_paths") as mock_tools:
        mock_tools.return_value = MagicMock(ffmpeg=Path("/usr/bin/ffmpeg"))
        
        cmd = step._build_command(
            Path("/usr/bin/ffmpeg"),
            audio_file,
            tmp_path / "output.mp4",
            positioned,
            162.0,  # duration
            1920,
            1080,
        )
        
        # Verify filter_complex_script is used (filter is in a file)
        assert "-filter_complex_script" in cmd
        filter_script_idx = cmd.index("-filter_complex_script")
        filter_file_path = Path(cmd[filter_script_idx + 1])
        assert filter_file_path.exists()
        filter_chain = filter_file_path.read_text()
        
        # Verify filter chain structure
        assert filter_chain.startswith("[1:v]")
        assert filter_chain.endswith("[out]")
        assert "drawtext=" in filter_chain
        
        # Count drawtext filters (should be 162)
        drawtext_count = filter_chain.count("drawtext=")
        assert drawtext_count == 162
        
        # Verify if expressions are present (using 'if' parameter)
        assert "if=" in filter_chain
        assert "between(t," in filter_chain
        
        # Verify semicolon-separated filter chains (new approach)
        assert ";" in filter_chain  # Should use semicolons to separate filters
        # Verify intermediate labels [v1], [v2], etc. are present
        assert "[v1]" in filter_chain
        # Last filter should output to [out]
        assert "[out]" in filter_chain
        
        # Verify output mapping
        assert "-map" in cmd
        map_out_idx = cmd.index("-map")
        assert cmd[map_out_idx + 1] == "[out]"
        
        # Verify audio mapping
        assert "0:a" in cmd or cmd[cmd.index("-map", map_out_idx + 1) + 1] == "0:a"
        
        # Log the filter chain length for reference
        print(f"\nFilter chain length: {len(filter_chain)} characters")
        print(f"Number of drawtext filters: {drawtext_count}")
        print(f"First 200 chars: {filter_chain[:200]}")
        print(f"Last 200 chars: {filter_chain[-200:]}")


def test_filter_chain_syntax_validity() -> None:
    """Test that filter chain syntax is valid FFmpeg syntax."""
    step = GenerateLyricVideoStep()
    
    # Create a few positioned words
    words = [
        PositionedWord(
            word=WordEntry(text="Hello", start_ms=1000, end_ms=2000),
            x=100,
            y=540,
            line=0,
        ),
        PositionedWord(
            word=WordEntry(text="world", start_ms=2000, end_ms=3000),
            x=200,
            y=540,
            line=0,
        ),
        PositionedWord(
            word=WordEntry(text="test", start_ms=3000, end_ms=4000),
            x=300,
            y=540,
            line=0,
        ),
    ]
    
    # Build filters
    filter_parts = []
    for pw in words:
        filter_str = step._build_drawtext_filter(pw, 1920, 1080)
        filter_parts.append(filter_str)
    
    # Build filter chain
    filter_chain = "[1:v]" + ",".join(filter_parts) + "[out]"
    
    # Verify structure
    assert filter_chain.startswith("[1:v]")
    assert filter_chain.endswith("[out]")
    
    # Verify each filter starts with drawtext=
    # Note: Can't simply split by comma because commas exist inside enable expressions
    # Instead, verify the structure by counting drawtext= occurrences
    drawtext_count = filter_chain.count("drawtext=")
    assert drawtext_count == len(words)
    
    # Verify if expressions have proper syntax (using 'if' parameter instead of 'enable')
    assert "if='between(t," in filter_chain
    
    print(f"\nSample filter chain: {filter_chain}")
