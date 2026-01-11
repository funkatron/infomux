# infomux

A local-first, deterministic media pipeline CLI.

## Philosophy

**infomux** is a tool, not an agent. It processes media files through a series of well-defined pipeline steps, producing derived artifacts (transcripts, summaries, images, drafts) in a predictable, reproducible manner.

### Core Principles

1. **Local-first**: All processing happens on your machine by default. Network operations (API calls, uploads) require explicit configuration and are never implicit.

2. **Deterministic**: Given the same inputs and configuration, infomux produces the same outputs. Random seeds and model versions are recorded in the job envelope.

3. **Auditable**: Every run creates a job envelope (`job.json`) that records inputs, outputs, parameters, and timing. You can inspect exactly what happened.

4. **Modular**: Each pipeline step is a small, testable module. Steps can be run independently or composed into pipelines.

5. **Boring**: The CLI is stable and predictable. stdout is reserved for machine-readable output; human-readable logs go to stderr.

### What infomux is NOT

- It is not an "AI agent" that makes decisions for you
- It does not perform destructive actions without explicit configuration
- It does not phone home or collect telemetry
- It does not anthropomorphize its operations

---

## Current Status

### ✅ Implemented

| Feature | Description |
|---------|-------------|
| **CLI scaffold** | `infomux run`, `inspect`, `resume` subcommands |
| **Job envelope** | Full JSON serialization with input hashing, step timing, artifact tracking |
| **Run storage** | `~/.local/share/infomux/runs/<run-id>/` with `job.json` |
| **extract_audio step** | ffmpeg wrapper → 16kHz mono PCM WAV |
| **transcribe step** | whisper-cli (whisper.cpp) → `transcript.txt` |
| **Dependency check** | `infomux run --check-deps` validates ffmpeg, whisper-cli, model |
| **Dry-run mode** | `infomux run --dry-run` outputs job envelope without executing |
| **Inspect command** | Human-readable and JSON output for completed runs |
| **Logging to stderr** | All logs to stderr; stdout reserved for machine output |

### 🚧 Partially Implemented

| Feature | Status |
|---------|--------|
| **resume command** | Parses args and loads job, but step re-execution not wired up |
| **Step selection** | `--steps` flag parses but only `extract_audio,transcribe` exist |

### ❌ Not Yet Implemented

| Feature | Notes |
|---------|-------|
| **Summarization** | Planned: Ollama integration (e.g., `qwen2.5:14b-instruct`) |
| **Frame extraction** | Planned: Extract key frames from video |
| **Custom pipelines** | Config file for defining step sequences |
| **Model auto-download** | Planned: `infomux setup` command |
| **Seed recording** | For full reproducibility of LLM outputs |
| **Progress reporting** | Real-time step progress to stderr |

---

## Requirements

### External Dependencies

| Tool | Install | Purpose |
|------|---------|---------|
| `ffmpeg` | `brew install ffmpeg` | Audio extraction |
| `whisper-cli` | `brew install whisper-cpp` | Local transcription |
| Whisper model | See below | GGML model file |

### Whisper Model Setup

```bash
# Create model directory
mkdir -p ~/.local/share/infomux/models/whisper

# Download base English model (~142 MB)
curl -L -o ~/.local/share/infomux/models/whisper/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# Set environment variable (add to ~/.zshrc)
export INFOMUX_WHISPER_MODEL="$HOME/.local/share/infomux/models/whisper/ggml-base.en.bin"
```

### Verify Setup

```bash
infomux run --check-deps
```

---

## Installation

```bash
# Clone and install with uv (recommended)
cd infomux
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Usage

### Run a Pipeline

```bash
# Process a media file (audio or video)
infomux run input.mp4

# Verbose output
infomux -v run input.mp4

# Dry run (show job envelope without executing)
infomux run --dry-run input.mp4

# Run specific steps only
infomux run --steps extract_audio input.mp4
```

### Inspect Runs

```bash
# List all runs
infomux inspect --list

# View run details (human-readable)
infomux inspect <run-id>

# View run details (JSON)
infomux inspect --json <run-id>
```

### Resume (Stub)

```bash
# Resume an interrupted run
infomux resume <run-id>
```

---

## Data Storage

Runs are stored under `~/.local/share/infomux/runs/` by default.

Each run creates a directory containing:

```
run-20260111-020549-c36c19/
├── job.json          # Job envelope with metadata
├── audio.wav         # Extracted audio (16kHz mono)
└── transcript.txt    # Transcription output
```

---

## Job Envelope

Every run produces a `job.json` file:

```json
{
  "id": "run-20260111-020549-c36c19",
  "created_at": "2026-01-11T02:05:49.359383+00:00",
  "updated_at": "2026-01-11T02:05:49.913183+00:00",
  "status": "completed",
  "input": {
    "path": "/path/to/input.mp4",
    "sha256": "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e",
    "size_bytes": 352078
  },
  "steps": [
    {
      "name": "extract_audio",
      "status": "completed",
      "started_at": "2026-01-11T02:05:49.362Z",
      "completed_at": "2026-01-11T02:05:49.551Z",
      "duration_seconds": 0.19,
      "outputs": ["audio.wav"]
    },
    {
      "name": "transcribe",
      "status": "completed",
      "started_at": "2026-01-11T02:05:49.551Z",
      "completed_at": "2026-01-11T02:05:49.912Z",
      "duration_seconds": 0.37,
      "outputs": ["transcript.txt"]
    }
  ],
  "artifacts": ["audio.wav", "transcript.txt"],
  "config": {},
  "error": null
}
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INFOMUX_DATA_DIR` | Base directory for runs | `~/.local/share/infomux` |
| `INFOMUX_LOG_LEVEL` | Log verbosity (DEBUG, INFO, WARN, ERROR) | `INFO` |
| `INFOMUX_WHISPER_MODEL` | Path to GGML whisper model | `~/.local/share/infomux/models/whisper/ggml-base.en.bin` |
| `INFOMUX_FFMPEG_PATH` | Override ffmpeg binary location | (auto-detected) |
| `INFOMUX_WHISPER_CLI_PATH` | Override whisper-cli binary location | (auto-detected) |

---

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest -v

# Lint
ruff check src/

# Type check (if mypy added)
# mypy src/
```

---

## License

MIT
