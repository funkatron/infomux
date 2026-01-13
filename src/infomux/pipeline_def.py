"""
Pipeline definition for infomux.

Defines pipelines as data structures that can be:
- Hardcoded (default pipeline)
- Loaded from config files (future)
- Passed via CLI (future)

This separates "what to run" from "how to run it".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepDef:
    """
    Definition of a pipeline step.

    Attributes:
        name: Step identifier (must match a registered step).
        input_from: Which step's output to use as input.
                   None = use original input file.
                   Step name = use that step's first output.
        config: Step-specific configuration.
    """

    name: str
    input_from: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "input_from": self.input_from,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepDef:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            input_from=data.get("input_from"),
            config=data.get("config", {}),
        )


@dataclass
class PipelineDef:
    """
    Definition of a complete pipeline.

    A pipeline is an ordered sequence of steps, where each step
    can specify which previous step's output it uses as input.

    Attributes:
        name: Human-readable pipeline name.
        description: What this pipeline does.
        steps: Ordered list of step definitions.
    """

    name: str
    description: str
    steps: list[StepDef]

    def step_names(self) -> list[str]:
        """Get ordered list of step names."""
        return [s.name for s in self.steps]

    def get_step(self, name: str) -> StepDef | None:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (for YAML/JSON config)."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineDef:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=[StepDef.from_dict(s) for s in data["steps"]],
        )


# =============================================================================
# Built-in Pipelines
# =============================================================================

DEFAULT_PIPELINE = PipelineDef(
    name="transcribe",
    description="Extract audio and transcribe to text",
    steps=[
        StepDef(
            name="extract_audio",
            input_from=None,  # Uses original input file
        ),
        StepDef(
            name="transcribe",
            input_from="extract_audio",  # Uses audio.wav from previous step
        ),
    ],
)

SUMMARIZE_PIPELINE = PipelineDef(
    name="summarize",
    description="Extract audio, transcribe, and summarize with LLM",
    steps=[
        StepDef(
            name="extract_audio",
            input_from=None,
        ),
        StepDef(
            name="transcribe",
            input_from="extract_audio",
        ),
        StepDef(
            name="summarize",
            input_from="transcribe",  # Uses transcript.txt
        ),
    ],
)

# Timed transcription: word-level timestamps without video embedding
TIMED_PIPELINE = PipelineDef(
    name="timed",
    description="Transcribe with word-level timestamps (SRT/VTT/JSON)",
    steps=[
        StepDef(
            name="extract_audio",
            input_from=None,
        ),
        StepDef(
            name="transcribe_timed",
            input_from="extract_audio",
        ),
    ],
)

# Caption pipeline: transcribe with timing and embed subtitles
CAPTION_PIPELINE = PipelineDef(
    name="caption",
    description="Transcribe with word-level timing and embed subtitles into video",
    steps=[
        StepDef(
            name="extract_audio",
            input_from=None,
        ),
        StepDef(
            name="transcribe_timed",
            input_from="extract_audio",
        ),
        StepDef(
            name="embed_subs",
            input_from=None,  # Uses original video; finds .srt in output_dir
            config={"burn_in": False},
        ),
    ],
)

# Caption with burn-in: permanently render subtitles into video
CAPTION_BURN_PIPELINE = PipelineDef(
    name="caption-burn",
    description="Transcribe and burn subtitles permanently into video",
    steps=[
        StepDef(
            name="extract_audio",
            input_from=None,
        ),
        StepDef(
            name="transcribe_timed",
            input_from="extract_audio",
        ),
        StepDef(
            name="embed_subs",
            input_from=None,
            config={"burn_in": True},  # Permanent subtitles
        ),
    ],
)

# Full report: transcript + timestamps + summary
REPORT_PIPELINE = PipelineDef(
    name="report",
    description="Full analysis: transcript, timestamps, and summary",
    steps=[
        StepDef(
            name="extract_audio",
            input_from=None,
        ),
        StepDef(
            name="transcribe",
            input_from="extract_audio",
        ),
        StepDef(
            name="transcribe_timed",
            input_from="extract_audio",  # Also uses audio.wav
        ),
        StepDef(
            name="summarize",
            input_from="transcribe",  # Uses transcript.txt
        ),
    ],
)

# Report with storage: full analysis saved to SQLite
REPORT_STORE_PIPELINE = PipelineDef(
    name="report-store",
    description="Full analysis stored in searchable SQLite database",
    steps=[
        StepDef(
            name="extract_audio",
            input_from=None,
        ),
        StepDef(
            name="transcribe",
            input_from="extract_audio",
        ),
        StepDef(
            name="transcribe_timed",
            input_from="extract_audio",
        ),
        StepDef(
            name="summarize",
            input_from="transcribe",
        ),
        StepDef(
            name="store_sqlite",
            input_from=None,  # Reads from run directory
        ),
    ],
)

# Generate video from audio with burned subtitles
AUDIO_TO_VIDEO_PIPELINE = PipelineDef(
    name="audio-to-video",
    description="Generate video from audio with burned-in subtitles",
    steps=[
        StepDef(
            name="extract_audio",
            input_from=None,
        ),
        StepDef(
            name="transcribe_timed",
            input_from="extract_audio",
        ),
        StepDef(
            name="generate_video",
            input_from="extract_audio",  # Uses audio.wav
            config={
                "background_color": "black",
                "video_size": "1920x1080",
            },
        ),
    ],
)

# Available pipelines
PIPELINES: dict[str, PipelineDef] = {
    "transcribe": DEFAULT_PIPELINE,
    "summarize": SUMMARIZE_PIPELINE,
    "timed": TIMED_PIPELINE,
    "caption": CAPTION_PIPELINE,
    "caption-burn": CAPTION_BURN_PIPELINE,
    "report": REPORT_PIPELINE,
    "report-store": REPORT_STORE_PIPELINE,
    "audio-to-video": AUDIO_TO_VIDEO_PIPELINE,
}


def get_pipeline(name: str | None = None) -> PipelineDef:
    """
    Get a pipeline by name.

    Args:
        name: Pipeline name. None returns the default.

    Returns:
        PipelineDef instance.

    Raises:
        ValueError: If pipeline not found.
    """
    if name is None:
        return DEFAULT_PIPELINE

    if name not in PIPELINES:
        available = ", ".join(PIPELINES.keys())
        raise ValueError(f"Unknown pipeline: {name}. Available: {available}")

    return PIPELINES[name]


def list_pipelines() -> list[str]:
    """List available pipeline names."""
    return list(PIPELINES.keys())
