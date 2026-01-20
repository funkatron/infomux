"""
The 'inspect' command: display details about a run.

This command shows the job envelope and status of a completed or
in-progress run. It's useful for debugging and auditing.

Usage:
    infomux inspect <run-id>
    infomux inspect --json <run-id>
    infomux inspect --list
    infomux inspect --list-pipelines
    infomux inspect --list-steps
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace

from infomux.log import get_logger
from infomux.pipeline_def import get_pipeline, list_pipelines
from infomux.steps import get_step, list_steps
from infomux.storage import list_runs, load_job, run_exists

logger = get_logger(__name__)


def configure_parser(parser: ArgumentParser) -> None:
    """
    Configure the argument parser for the 'inspect' command.

    Args:
        parser: The subparser to configure.
    """
    parser.add_argument(
        "run_id",
        type=str,
        nargs="?",
        default=None,
        help="ID of the run to inspect",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (default: human-readable)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        dest="list_runs",
        help="List all runs instead of inspecting one",
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List available pipelines and exit",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List available steps and exit",
    )


def execute(args: Namespace) -> int:
    """
    Execute the 'inspect' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # List pipelines mode
    if args.list_pipelines:
        return _list_pipelines()

    # List steps mode
    if args.list_steps:
        return _list_steps()

    # List runs mode
    if args.list_runs:
        return _list_runs(args.json)

    # Inspect mode requires a run ID
    if not args.run_id:
        logger.error("run_id is required (or use --list, --list-pipelines, or --list-steps)")
        return 1

    run_id = args.run_id

    # Check if run exists
    if not run_exists(run_id):
        logger.error("run not found: %s", run_id)
        return 1

    # Load the job envelope
    try:
        job = load_job(run_id)
    except Exception as e:
        logger.error("failed to load run: %s", e)
        return 1

    # Output
    if args.json:
        print(job.to_json(), file=sys.stdout)
    else:
        # Human-readable format
        _print_job_summary(job)

    return 0


def _list_runs(output_json: bool = False) -> int:
    """
    List all runs with summary information in a tabular format.

    Args:
        output_json: If True, output as JSON array.

    Returns:
        Exit code (0 for success).
    """
    runs = list_runs()
    if not runs:
        if output_json:
            print("[]", file=sys.stdout)
        else:
            print("No runs found.")
        return 0

    if output_json:
        # JSON output: array of run summaries
        import json
        summaries = []
        for run_id in runs:
            try:
                job = load_job(run_id)
                summary = {
                    "id": job.id,
                    "status": job.status,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                }
                if job.config.get("pipeline"):
                    summary["pipeline"] = job.config["pipeline"]
                if job.input:
                    summary["input"] = {
                        "path": job.input.path,
                        "size_bytes": job.input.size_bytes,
                    }
                    if job.input.original_url:
                        summary["input"]["original_url"] = job.input.original_url
                if job.artifacts:
                    summary["artifacts"] = job.artifacts
                summaries.append(summary)
            except Exception:
                # Skip runs that can't be loaded
                continue
        print(json.dumps(summaries, indent=2), file=sys.stdout)
    else:
        # Tabular output
        rows = []
        for run_id in runs:
            try:
                job = load_job(run_id)
                
                # Status icon
                status_icon = {
                    "pending": "○",
                    "running": "◐",
                    "completed": "●",
                    "failed": "✗",
                    "interrupted": "⚠",
                }.get(job.status, "?")
                
                # Format date (just date part, not full timestamp)
                created_date = job.created_at.split("T")[0] if "T" in job.created_at else job.created_at[:10]
                
                # Pipeline name
                pipeline_name = job.config.get("pipeline", "?")
                
                # Input file name
                input_name = "?"
                if job.input:
                    from pathlib import Path
                    input_path = Path(job.input.path)
                    input_name = input_path.name
                    if job.input.original_url:
                        # Show URL for downloaded files
                        from urllib.parse import urlparse
                        parsed = urlparse(job.input.original_url)
                        input_name = f"{parsed.netloc}...{input_path.suffix}" if parsed.netloc else input_name
                
                # Artifact count
                artifact_count = len(job.artifacts) if job.artifacts else 0
                
                rows.append({
                    "icon": status_icon,
                    "id": run_id,
                    "status": job.status,
                    "date": created_date,
                    "pipeline": pipeline_name,
                    "input": input_name,
                    "artifacts": artifact_count,
                })
            except Exception as e:
                # Show run ID even if we can't load details
                logger.debug("failed to load run %s: %s", run_id, e)
                rows.append({
                    "icon": "?",
                    "id": run_id,
                    "status": "error",
                    "date": "?",
                    "pipeline": "?",
                    "input": "?",
                    "artifacts": 0,
                })
        
        # Print table
        if rows:
            # Calculate column widths
            max_id_len = max(len(r["id"]) for r in rows)
            max_status_len = max(len(r["status"]) for r in rows)
            max_pipeline_len = max(len(r["pipeline"]) for r in rows)
            max_input_len = max(len(r["input"]) for r in rows)
            
            # Ensure minimum widths for headers
            max_id_len = max(max_id_len, len("Run ID"))
            max_status_len = max(max_status_len, len("Status"))
            max_pipeline_len = max(max_pipeline_len, len("Pipeline"))
            max_input_len = max(max_input_len, len("Input"))
            
            # Header
            header = (
                f"{'':2} "
                f"{'Run ID':<{max_id_len}} "
                f"{'Status':<{max_status_len}} "
                f"{'Date':<10} "
                f"{'Pipeline':<{max_pipeline_len}} "
                f"{'Input':<{max_input_len}} "
                f"{'Artifacts':>9}"
            )
            print(header)
            print("-" * len(header))
            
            # Rows
            for row in rows:
                artifacts_str = str(row["artifacts"]) if row["artifacts"] > 0 else "-"
                print(
                    f"{row['icon']:2} "
                    f"{row['id']:<{max_id_len}} "
                    f"{row['status']:<{max_status_len}} "
                    f"{row['date']:<10} "
                    f"{row['pipeline']:<{max_pipeline_len}} "
                    f"{row['input']:<{max_input_len}} "
                    f"{artifacts_str:>9}"
                )
            print()
            print(f"Total: {len(rows)} run(s)")

    return 0


def _list_pipelines() -> int:
    """
    List available pipelines.

    Returns:
        Exit code (always 0).
    """
    print("Available pipelines:")
    print()
    for name in list_pipelines():
        pipeline = get_pipeline(name)
        print(f"  {name}")
        print(f"    {pipeline.description}")
        print(f"    Steps: {' → '.join(pipeline.step_names())}")
        print()
    return 0


def _list_steps() -> int:
    """
    List available steps.

    Returns:
        Exit code (always 0).
    """
    print("Available steps:")
    print()
    steps = sorted(list_steps())
    for name in steps:
        step_info = get_step(name)
        if step_info:
            output_info = ""
            if step_info.output_filename:
                output_info = f" → {step_info.output_filename}"
            module_info = ""
            if step_info.module:
                # Extract just the module name (e.g., "extract_audio" from "infomux.steps.extract_audio")
                module_name = step_info.module.split(".")[-1] if "." in step_info.module else step_info.module
                module_info = f" ({module_name})"
            print(f"  {name}{module_info}{output_info}")
        else:
            print(f"  {name}")
    print()
    print(f"Total: {len(steps)} step(s)")
    return 0


def _print_job_summary(job) -> None:
    """
    Print a human-readable summary of a job.

    Args:
        job: The JobEnvelope to summarize.
    """
    print(f"Run: {job.id}")
    print(f"Status: {job.status}")
    print(f"Created: {job.created_at}")
    print(f"Updated: {job.updated_at}")

    if job.input:
        print("\nInput:")
        print(f"  Path: {job.input.path}")
        print(f"  SHA256: {job.input.sha256[:16]}...")
        print(f"  Size: {job.input.size_bytes:,} bytes")

    if job.steps:
        print("\nSteps:")
        for step in job.steps:
            status_icon = {
                "pending": "○",
                "running": "◐",
                "completed": "●",
                "failed": "✗",
            }.get(step.status, "?")
            print(f"  {status_icon} {step.name}: {step.status}")
            if step.duration_seconds:
                print(f"      Duration: {step.duration_seconds:.2f}s")
            if step.error:
                print(f"      Error: {step.error}")

    if job.artifacts:
        print("\nArtifacts:")
        for artifact in job.artifacts:
            print(f"  - {artifact}")

    if job.error:
        print(f"\nError: {job.error}")
