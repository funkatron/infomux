# infomux

A local-first, deterministic media pipeline CLI.

```
infomux run podcast.mp4
# → ~/.local/share/infomux/runs/run-20260111-020549-c36c19/
#   ├── job.json
#   ├── audio.wav
#   └── transcript.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
brew install ffmpeg whisper-cpp

# 2. Download whisper model (~142 MB)
mkdir -p ~/.local/share/infomux/models/whisper
curl -L -o ~/.local/share/infomux/models/whisper/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# 3. Set model path (add to ~/.zshrc)
export INFOMUX_WHISPER_MODEL="$HOME/.local/share/infomux/models/whisper/ggml-base.en.bin"

# 4. Install infomux
cd infomux
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .

# 5. Verify setup
infomux run --check-deps

# 6. Run on a media file
infomux run your-file.mp4
```

---

## Philosophy

**infomux** is a tool, not an agent.

It processes media files through well-defined pipeline steps, producing derived artifacts (transcripts, summaries, images) in a predictable, reproducible manner.

| Principle | What it means |
|-----------|---------------|
| **Local-first** | All processing on your machine. No implicit network calls. |
| **Deterministic** | Same inputs → same outputs. Seeds and versions recorded. |
| **Auditable** | Every run creates `job.json` with full execution trace. |
| **Modular** | Each step is small, testable, composable. |
| **Boring** | Stable CLI. stdout = machine output, stderr = logs. |

### What infomux is NOT

- Not an "AI agent" that makes autonomous decisions
- No destructive actions without explicit configuration
- No telemetry or phoning home
- No anthropomorphic language in code or output

---

## Commands

### `infomux run`

Process a media file through a pipeline.

```bash
# Basic usage (uses default 'transcribe' pipeline)
infomux run input.mp4

# Verbose logging
infomux -v run input.mp4

# Specify a pipeline
infomux run --pipeline transcribe input.mp4

# List available pipelines
infomux run --list-pipelines

# Run specific steps only (subset of pipeline)
infomux run --steps extract_audio input.mp4

# Preview without executing
infomux run --dry-run input.mp4

# Check dependencies
infomux run --check-deps
```

**Output:** Prints the run directory path to stdout.

### `infomux inspect`

View details of a completed run.

```bash
# List all runs (most recent first)
infomux inspect --list

# Human-readable summary
infomux inspect run-20260111-020549-c36c19

# JSON output (for scripting)
infomux inspect --json run-20260111-020549-c36c19
```

**Example output:**

```
Run: run-20260111-020549-c36c19
Status: completed
Created: 2026-01-11T02:05:49+00:00
Updated: 2026-01-11T02:05:49+00:00

Input:
  Path: /path/to/input.mp4
  SHA256: 59dfb9a4acb36fe2...
  Size: 352,078 bytes

Steps:
  ● extract_audio: completed
      Duration: 0.19s
  ● transcribe: completed
      Duration: 0.37s

Artifacts:
  - audio.wav
  - transcript.txt
```

### `infomux resume`

Resume an interrupted or failed run, or re-run specific steps.

```bash
# Resume from first incomplete step
infomux resume <run-id>

# Re-run from a specific step (and all following)
infomux resume --from-step transcribe <run-id>

# Preview what would run
infomux resume --dry-run <run-id>
```

**Behavior:**
- Loads existing job envelope from the run directory
- Skips already-completed steps (unless `--from-step` specified)
- Clears failed step records before re-running
- Uses the same pipeline and input as the original run

---

## Pipeline Steps

| Step | Input | Output | Tool |
|------|-------|--------|------|
| `extract_audio` | media file | `audio.wav` (16kHz mono) | ffmpeg |
| `transcribe` | `audio.wav` | `transcript.txt` | whisper-cli |

### Default Pipeline

```
input.mp4 → [extract_audio] → audio.wav → [transcribe] → transcript.txt
```

---

## Data Storage

### Run Directory

Each run creates a directory under `~/.local/share/infomux/runs/`:

```
~/.local/share/infomux/
├── runs/
│   ├── run-20260111-020549-c36c19/
│   │   ├── job.json          # Execution metadata
│   │   ├── audio.wav         # Extracted audio
│   │   └── transcript.txt    # Transcription
│   └── run-20260111-021000-ab12cd/
│       └── ...
└── models/
    └── whisper/
        └── ggml-base.en.bin  # Whisper model
```

### Job Envelope (`job.json`)

Every run produces a complete execution record:

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

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INFOMUX_DATA_DIR` | Base directory for runs and models | `~/.local/share/infomux` |
| `INFOMUX_LOG_LEVEL` | Log verbosity: `DEBUG`, `INFO`, `WARN`, `ERROR` | `INFO` |
| `INFOMUX_WHISPER_MODEL` | Path to GGML whisper model file | `$INFOMUX_DATA_DIR/models/whisper/ggml-base.en.bin` |
| `INFOMUX_FFMPEG_PATH` | Override ffmpeg binary location | *(auto-detected from PATH)* |
| `INFOMUX_WHISPER_CLI_PATH` | Override whisper-cli binary location | *(auto-detected from PATH)* |

### Whisper Model Options

| Model | Size | Speed | Quality | Download |
|-------|------|-------|---------|----------|
| `ggml-tiny.en.bin` | 75 MB | Fastest | Basic | [link](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin) |
| `ggml-base.en.bin` | 142 MB | Fast | Good | [link](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin) |
| `ggml-small.en.bin` | 466 MB | Medium | Better | [link](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin) |
| `ggml-medium.en.bin` | 1.5 GB | Slow | Best | [link](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin) |

---

## Troubleshooting

### `ffmpeg not found`

```bash
brew install ffmpeg
```

### `whisper-cli not found`

```bash
brew install whisper-cpp
```

> ⚠️ **Note:** Use `whisper-cli` (from `whisper-cpp`), NOT the Python `whisper` package.

### `Whisper model not found`

```bash
mkdir -p ~/.local/share/infomux/models/whisper
curl -L -o ~/.local/share/infomux/models/whisper/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
export INFOMUX_WHISPER_MODEL="$HOME/.local/share/infomux/models/whisper/ggml-base.en.bin"
```

### `Metal acceleration not working` (Apple Silicon)

whisper-cpp from Homebrew includes Metal support. If transcription is slow, ensure you're using the Homebrew version:

```bash
which whisper-cli
# Should show: /opt/homebrew/bin/whisper-cli
```

---

## Project Structure

```
src/infomux/
├── __init__.py         # Package version
├── __main__.py         # python -m infomux entry
├── cli.py              # Argument parsing and subcommand dispatch
├── config.py           # Tool paths and environment variables
├── job.py              # JobEnvelope, InputFile, StepRecord dataclasses
├── log.py              # Logging configuration (stderr only)
├── pipeline.py         # Step orchestration
├── storage.py          # Run directory management
├── commands/
│   ├── run.py          # infomux run
│   ├── inspect.py      # infomux inspect
│   └── resume.py       # infomux resume
└── steps/
    ├── __init__.py     # Step protocol and registry
    ├── extract_audio.py # ffmpeg wrapper
    └── transcribe.py   # whisper-cli wrapper
```

---

## Implementation Status

### ✅ Implemented

- CLI scaffold with `run`, `inspect`, `resume` subcommands
- Job envelope with input hashing, step timing, artifact tracking
- Run storage under `~/.local/share/infomux/runs/`
- `extract_audio` step (ffmpeg → 16kHz mono WAV)
- `transcribe` step (whisper-cli → transcript.txt)
- Pipeline definitions as data (`PipelineDef`, `StepDef`)
- Step input/output dependency resolution
- `--pipeline` and `--list-pipelines` flags
- `resume` command with `--from-step` support
- Dependency checking (`--check-deps`)
- Dry-run mode (`--dry-run`)
- Logging to stderr

### 🚧 Partially Implemented

- `--steps` flag (parses, validates, but only 2 steps exist)

### ❌ Planned

- **Summarization** — Ollama integration
- **Frame extraction** — Key frames from video
- **Custom pipelines** — Load from YAML/JSON config file
- **Model auto-download** — `infomux setup` command
- **Seed recording** — Full LLM reproducibility
- **Progress reporting** — Real-time step progress

---

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest -v

# Lint
ruff check src/

# Format
ruff format src/
```

---

## License

MIT
