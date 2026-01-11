"""
Store step: upload run artifacts to S3.

Uploads run directory contents to an S3 bucket for cloud backup
and sharing. Requires AWS credentials and boto3.

Environment variables:
    INFOMUX_S3_BUCKET: Target S3 bucket name (required)
    INFOMUX_S3_PREFIX: Key prefix (default: "infomux/")
    AWS_PROFILE / AWS_ACCESS_KEY_ID / etc: AWS credentials

Output: No local files, uploads to S3
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from infomux.log import get_logger
from infomux.steps import StepError, StepResult, register_step

logger = get_logger(__name__)

# No local output file
STORE_S3_FILENAME = None


def _get_s3_config() -> tuple[str, str]:
    """
    Get S3 configuration from environment.

    Returns:
        Tuple of (bucket_name, prefix).

    Raises:
        StepError: If bucket not configured.
    """
    bucket = os.environ.get("INFOMUX_S3_BUCKET")
    if not bucket:
        raise StepError(
            "store_s3",
            "INFOMUX_S3_BUCKET not set. Set to your S3 bucket name.",
        )

    prefix = os.environ.get("INFOMUX_S3_PREFIX", "infomux/")
    if not prefix.endswith("/"):
        prefix += "/"

    return bucket, prefix


def _upload_to_s3(
    run_dir: Path,
    bucket: str,
    prefix: str,
) -> list[str]:
    """
    Upload run directory to S3.

    Args:
        run_dir: Local run directory.
        bucket: S3 bucket name.
        prefix: S3 key prefix.

    Returns:
        List of uploaded S3 keys.

    Raises:
        StepError: If upload fails.
    """
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        raise StepError(
            "store_s3",
            "boto3 not installed. Run: pip install boto3",
        )

    s3 = boto3.client("s3")
    run_id = run_dir.name
    uploaded = []

    try:
        for file_path in run_dir.iterdir():
            if file_path.is_file():
                key = f"{prefix}{run_id}/{file_path.name}"

                logger.debug("uploading: %s -> s3://%s/%s", file_path.name, bucket, key)

                s3.upload_file(
                    str(file_path),
                    bucket,
                    key,
                    ExtraArgs={"ContentType": _get_content_type(file_path)},
                )
                uploaded.append(key)

        return uploaded

    except (BotoCoreError, ClientError) as e:
        raise StepError("store_s3", f"S3 upload failed: {e}")


def _get_content_type(path: Path) -> str:
    """Get content type for file."""
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
        ".srt": "text/plain",
        ".vtt": "text/vtt",
    }.get(suffix, "application/octet-stream")


@register_step
@dataclass
class StoreS3Step:
    """
    Pipeline step to upload run artifacts to S3.

    Uploads all files in the run directory to:
        s3://{bucket}/{prefix}{run_id}/

    Requires:
    - boto3 installed
    - AWS credentials configured
    - INFOMUX_S3_BUCKET environment variable
    """

    name: str = "store_s3"

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Upload run directory to S3.

        Args:
            input_path: Not used.
            output_dir: The run directory to upload.

        Returns:
            Empty list (no local outputs).

        Raises:
            StepError: If upload fails.
        """
        bucket, prefix = _get_s3_config()

        logger.info("uploading to s3://%s/%s%s/", bucket, prefix, output_dir.name)

        uploaded = _upload_to_s3(output_dir, bucket, prefix)

        logger.info("uploaded %d files to S3", len(uploaded))
        return []


def run(input_path: Path, output_dir: Path) -> StepResult:
    """Run the store_s3 step."""
    step = StoreS3Step()
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
