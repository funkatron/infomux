"""
Store step: export run data as structured JSON.

Creates a single JSON file with all run data for external tools,
archival, or data exchange.

Output: report.json in run directory
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step
from infomux.steps.storage import RunData

logger = get_logger(__name__)

# Output filename
REPORT_JSON_FILENAME = "report.json"


@register_step
@dataclass
class StoreJsonStep:
    """
    Pipeline step to export run data as JSON.

    Creates a comprehensive JSON file suitable for:
    - Data exchange with external tools
    - Archival and backup
    - Custom processing pipelines
    """

    name: str = "store_json"

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Export run data as JSON.

        Args:
            input_path: Not used (reads from output_dir).
            output_dir: The run directory.

        Returns:
            List containing path to report.json.

        Raises:
            StepError: If export fails.
        """
        run_data = RunData.from_run_dir(output_dir)
        if not run_data:
            raise StepError(self.name, "No run data found")

        output_path = output_dir / REPORT_JSON_FILENAME

        logger.info("exporting JSON: %s", output_path.name)

        try:
            # Include artifacts from job.json
            artifacts = run_data.job_json.get("artifacts", [])
            
            export = {
                "version": "1.0",
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "run": run_data.to_dict(),
                "artifacts": artifacts,  # Include all artifacts (video files, etc.)
            }

            with open(output_path, "w") as f:
                json.dump(export, f, indent=2, ensure_ascii=False)

            size = output_path.stat().st_size
            logger.info("exported: %s (%d bytes)", output_path.name, size)
            return [output_path]

        except (OSError, TypeError) as e:
            raise StepError(self.name, f"export failed: {e}")


def run(input_path: Path, output_dir: Path) -> StepResult:
    """Run the store_json step."""
    step = StoreJsonStep()
    start_time = time.monotonic()

    try:
        outputs = step.execute(input_path, output_dir)
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=True,
            outputs=outputs,
            duration_seconds=duration,
        )
    except StepError as e:
        duration = time.monotonic() - start_time
        return StepResult(
            name=step.name,
            success=False,
            outputs=[],
            duration_seconds=duration,
            error=str(e),
        )
