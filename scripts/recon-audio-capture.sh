#!/usr/bin/env bash
# Thin wrapper: real logic lives in `infomux audio-recon`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec uv run infomux audio-recon "$@"
