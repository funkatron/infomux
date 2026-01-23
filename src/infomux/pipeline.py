"""
Pipeline orchestration for infomux.

Coordinates the execution of pipeline steps, updating the job envelope
as each step completes. Handles step dependencies and failure recovery.

Steps are auto-discovered from infomux.steps - no manual imports needed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from infomux.job import JobEnvelope, JobStatus, StepRecord
from infomux.log import get_logger
from infomux.pipeline_def import DEFAULT_PIPELINE, PipelineDef, get_pipeline
from infomux.steps import get_step_output, run_step

logger = get_logger(__name__)


def run_pipeline(
    job: JobEnvelope,
    run_dir: Path,
    pipeline: PipelineDef | None = None,
    step_names: list[str] | None = None,
    step_configs: dict[str, dict] | None = None,
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
            expected_output = get_step_output(step_def.input_from)
            if expected_output:
                input_path = run_dir / expected_output
                if not input_path.exists():
                    input_path = original_input
            else:
                input_path = original_input

        logger.debug("step input: %s", input_path)

        # Merge step config (from pipeline def) with CLI overrides
        step_config = dict(step_def.config) if step_def.config else {}
        if step_configs and step_name in step_configs:
            step_config.update(step_configs[step_name])
            logger.debug("merged config for %s: %s", step_name, step_config)

        # Create step record
        step_record = StepRecord(
            name=step_name,
            status="running",
            started_at=datetime.now(UTC).isoformat(),
        )
        job.steps.append(step_record)
        
        # Update job status to running if not already
        if job.status == JobStatus.PENDING.value:
            job.update_status(JobStatus.RUNNING)
        
        # Save job so running status is visible for inspection
        from infomux.storage import save_job
        save_job(job)

        # Run the step via registry
        result = run_step(step_name, input_path, run_dir, step_config if step_config else None)

        # Update step record
        step_record.status = "completed" if result.success else "failed"
        step_record.completed_at = datetime.now(UTC).isoformat()
        step_record.duration_seconds = result.duration_seconds
        step_record.outputs = [str(p) for p in result.outputs]
        step_record.model_info = result.model_info  # For LLM steps

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
