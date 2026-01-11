"""
Pipeline orchestration for infomux.

Coordinates the execution of pipeline steps, updating the job envelope
as each step completes. Handles step dependencies and failure recovery.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from infomux.job import JobEnvelope, JobStatus, StepRecord
from infomux.log import get_logger
from infomux.pipeline_def import DEFAULT_PIPELINE, PipelineDef, get_pipeline
from infomux.steps import StepResult
from infomux.steps.extract_audio import AUDIO_FILENAME
from infomux.steps.extract_audio import run as run_extract_audio
from infomux.steps.transcribe import TRANSCRIPT_FILENAME
from infomux.steps.transcribe import run as run_transcribe

logger = get_logger(__name__)

# Step output filenames for input resolution
STEP_OUTPUTS: dict[str, str] = {
    "extract_audio": AUDIO_FILENAME,
    "transcribe": TRANSCRIPT_FILENAME,
}


def run_pipeline(
    job: JobEnvelope,
    run_dir: Path,
    pipeline: PipelineDef | None = None,
    step_names: list[str] | None = None,
) -> bool:
    """
    Execute the pipeline steps for a job.

    Args:
        job: The job envelope to update with execution status.
        run_dir: Directory for run artifacts.
        pipeline: Pipeline definition to use (default: DEFAULT_PIPELINE).
        step_names: Optional subset of steps to run.

    Returns:
        True if all steps completed successfully, False otherwise.
    """
    if pipeline is None:
        pipeline = DEFAULT_PIPELINE

    if not job.input:
        logger.error("no input file specified in job")
        job.update_status(JobStatus.FAILED, "no input file specified")
        return False

    original_input = Path(job.input.path)

    # Determine which steps to run
    steps_to_run = pipeline.steps
    if step_names:
        steps_to_run = [s for s in pipeline.steps if s.name in step_names]
        if not steps_to_run:
            logger.error("no matching steps found")
            return False

    logger.info(
        "running pipeline '%s' with %d steps: %s",
        pipeline.name,
        len(steps_to_run),
        [s.name for s in steps_to_run],
    )

    # Store pipeline info in job config
    job.config["pipeline"] = pipeline.name
    job.config["pipeline_steps"] = pipeline.step_names()

    # Track outputs from each step for input resolution
    step_outputs: dict[str, Path] = {}
    all_success = True

    for step_def in steps_to_run:
        step_name = step_def.name
        logger.info("starting step: %s", step_name)

        # Resolve input for this step
        if step_def.input_from is None:
            input_path = original_input
        elif step_def.input_from in step_outputs:
            input_path = step_outputs[step_def.input_from]
        else:
            # Look for expected output file from previous step
            expected_output = STEP_OUTPUTS.get(step_def.input_from)
            if expected_output:
                input_path = run_dir / expected_output
                if not input_path.exists():
                    input_path = original_input
            else:
                input_path = original_input

        logger.debug("step input: %s", input_path)

        # Create step record
        step_record = StepRecord(
            name=step_name,
            status="running",
            started_at=datetime.now(UTC).isoformat(),
        )
        job.steps.append(step_record)

        # Run the step
        result = _run_step(step_name, input_path, run_dir, step_def.config)

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

        # Record outputs for subsequent steps
        if result.outputs:
            step_outputs[step_name] = result.outputs[0]

        # Add outputs to job artifacts
        job.artifacts.extend(step_record.outputs)

        logger.info(
            "step completed: %s (%.2fs, %d outputs)",
            step_name,
            result.duration_seconds,
            len(result.outputs),
        )

    return all_success


def _run_step(
    step_name: str,
    input_path: Path,
    output_dir: Path,
    config: dict | None = None,
) -> StepResult:
    """
    Run a single pipeline step.

    Args:
        step_name: Name of the step to run.
        input_path: Input file for this step.
        output_dir: Directory for output artifacts.
        config: Step-specific configuration.

    Returns:
        StepResult with execution details.
    """
    # Step dispatch - this could be made more dynamic with a registry
    if step_name == "extract_audio":
        return run_extract_audio(input_path, output_dir)
    elif step_name == "transcribe":
        return run_transcribe(input_path, output_dir)
    else:
        logger.error("unknown step: %s", step_name)
        return StepResult(
            name=step_name,
            success=False,
            outputs=[],
            duration_seconds=0,
            error=f"unknown step: {step_name}",
        )


def get_resumable_steps(
    job: JobEnvelope,
    from_step: str | None = None,
) -> list[str]:
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

    # Get pipeline from job config or use default
    pipeline_name = job.config.get("pipeline")
    pipeline = get_pipeline(pipeline_name)
    all_steps = pipeline.step_names()

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
