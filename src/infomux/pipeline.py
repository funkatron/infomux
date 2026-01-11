"""
Pipeline orchestration for infomux.

Coordinates the execution of pipeline steps, updating the job envelope
as each step completes. Handles step dependencies and failure recovery.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from infomux.job import JobEnvelope, JobStatus, StepRecord
from infomux.log import get_logger
from infomux.steps import StepResult
from infomux.steps.extract_audio import AUDIO_FILENAME
from infomux.steps.extract_audio import run as run_extract_audio
from infomux.steps.transcribe import run as run_transcribe

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Default pipeline: extract audio, then transcribe
DEFAULT_STEPS = ["extract_audio", "transcribe"]


def run_pipeline(
    job: JobEnvelope,
    run_dir: Path,
    steps: list[str] | None = None,
) -> bool:
    """
    Execute the pipeline steps for a job.

    Args:
        job: The job envelope to update with execution status.
        run_dir: Directory for run artifacts.
        steps: List of step names to run (default: all steps).

    Returns:
        True if all steps completed successfully, False otherwise.
    """
    if steps is None:
        steps = DEFAULT_STEPS

    if not job.input:
        logger.error("no input file specified in job")
        job.update_status(JobStatus.FAILED, "no input file specified")
        return False

    input_path = Path(job.input.path)

    logger.info("running pipeline with %d steps: %s", len(steps), steps)

    # Track the current input for each step
    # First step uses the original input, subsequent steps use outputs
    current_input = input_path
    all_success = True

    for step_name in steps:
        logger.info("starting step: %s", step_name)

        # Create step record
        step_record = StepRecord(
            name=step_name,
            status="running",
            started_at=datetime.now(UTC).isoformat(),
        )
        job.steps.append(step_record)

        # Run the step
        result = _run_step(step_name, current_input, run_dir)

        # Update step record
        step_record.status = "completed" if result.success else "failed"
        step_record.completed_at = datetime.now(UTC).isoformat()
        step_record.duration_seconds = result.duration_seconds
        step_record.outputs = [str(p) for p in result.outputs]

        if not result.success:
            step_record.error = result.error
            logger.error("step failed: %s - %s", step_name, result.error)
            job.update_status(
                JobStatus.FAILED, f"step '{step_name}' failed: {result.error}"
            )
            all_success = False
            break

        # Add outputs to job artifacts
        job.artifacts.extend(step_record.outputs)

        logger.info(
            "step completed: %s (%.2fs, %d outputs)",
            step_name,
            result.duration_seconds,
            len(result.outputs),
        )

        # Update current input for next step (use first output)
        if result.outputs:
            current_input = result.outputs[0]

    return all_success


def _run_step(step_name: str, input_path: Path, output_dir: Path) -> StepResult:
    """
    Run a single pipeline step.

    Args:
        step_name: Name of the step to run.
        input_path: Input file for this step.
        output_dir: Directory for output artifacts.

    Returns:
        StepResult with execution details.
    """
    if step_name == "extract_audio":
        return run_extract_audio(input_path, output_dir)
    elif step_name == "transcribe":
        # Transcribe step expects audio.wav as input
        audio_path = output_dir / AUDIO_FILENAME
        if audio_path.exists():
            input_path = audio_path
        return run_transcribe(input_path, output_dir)
    else:
        # Unknown step
        logger.error("unknown step: %s", step_name)
        return StepResult(
            name=step_name,
            success=False,
            outputs=[],
            duration_seconds=0,
            error=f"unknown step: {step_name}",
        )


def get_resumable_steps(job: JobEnvelope, from_step: str | None = None) -> list[str]:
    """
    Determine which steps need to be run for resumption.

    Args:
        job: The job envelope to analyze.
        from_step: Optional step name to resume from.

    Returns:
        List of step names to execute.
    """
    # Get completed steps
    completed = {s.name for s in job.steps if s.status == "completed"}

    # Get requested steps from config or use defaults
    all_steps = job.config.get("requested_steps", DEFAULT_STEPS)

    if from_step:
        # Resume from a specific step (re-run it and all following)
        try:
            idx = all_steps.index(from_step)
            return all_steps[idx:]
        except ValueError:
            logger.warning("step not found in pipeline: %s", from_step)
            return []

    # Resume from first incomplete step
    return [s for s in all_steps if s not in completed]
