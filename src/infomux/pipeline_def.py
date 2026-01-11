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

# Future pipelines can be defined here or loaded from config
PIPELINES: dict[str, PipelineDef] = {
    "transcribe": DEFAULT_PIPELINE,
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
