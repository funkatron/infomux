"""
Pipeline steps for infomux.

Each step is a module that:
- Receives an input (file path or artifact from previous step)
- Produces one or more output artifacts
- Records its execution in the job envelope

Steps are designed to be:
- Small and focused on one task
- Testable in isolation
- Deterministic (same inputs → same outputs)

Available steps (stubs - not yet implemented):
- extract_audio: Extract audio track from video
- transcribe: Transcribe audio to text
- summarize: Generate summary from transcript
- extract_frames: Extract key frames from video
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class StepProtocol(Protocol):
    """
    Protocol defining the interface for pipeline steps.

    All steps must implement this interface to be usable in the pipeline.
    """

    name: str

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Execute the step.

        Args:
            input_path: Path to the input file or artifact.
            output_dir: Directory where output artifacts should be written.

        Returns:
            List of paths to output artifacts produced by this step.

        Raises:
            StepError: If the step fails.
        """
        ...


@dataclass
class StepResult:
    """
    Result of executing a pipeline step.

    Attributes:
        name: Name of the step.
        success: Whether the step completed successfully.
        outputs: List of output artifact paths.
        duration_seconds: How long the step took.
        error: Error message if the step failed.
    """

    name: str
    success: bool
    outputs: list[Path]
    duration_seconds: float
    error: str | None = None


class StepError(Exception):
    """
    Exception raised when a pipeline step fails.

    Attributes:
        step_name: Name of the step that failed.
        message: Error message.
    """

    def __init__(self, step_name: str, message: str) -> None:
        self.step_name = step_name
        self.message = message
        super().__init__(f"Step '{step_name}' failed: {message}")


# Registry of available steps (populated as steps are implemented)
STEP_REGISTRY: dict[str, type] = {}


def register_step(step_class: type) -> type:
    """
    Decorator to register a step class in the registry.

    Usage:
        @register_step
        class MyStep:
            name = "my_step"
            ...

    Args:
        step_class: The step class to register.

    Returns:
        The step class (unchanged).
    """
    if hasattr(step_class, "name"):
        STEP_REGISTRY[step_class.name] = step_class
    return step_class


def get_step(name: str) -> type | None:
    """
    Get a step class by name.

    Args:
        name: The step name.

    Returns:
        The step class, or None if not found.
    """
    return STEP_REGISTRY.get(name)


def list_steps() -> list[str]:
    """
    List all registered step names.

    Returns:
        List of step names.
    """
    return list(STEP_REGISTRY.keys())
