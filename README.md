# infomux

A local-first CLI for transcribing audio/video and capturing voice notes.

**What it does:**
- Transcribe any audio/video file to text
- Record voice notes from your microphone with live transcription
- Generate summaries using local LLMs (Ollama)
- Keep everything on your machine — no cloud, no API keys

```bash
# Transcribe a podcast
infomux run podcast.mp4
# → transcript.txt

# Record a voice note
infomux stream
# → audio.wav + transcript with word-level timestamps

# Summarize a meeting
infomux run --pipeline summarize meeting.mp4
# → transcript.txt + summary.md
```

---

## Requirements

- **macOS** (tested) or **Linux** (should work, see notes)
- **Python 3.11+**
- **ffmpeg** and **whisper-cpp** (whisper.cpp)

### Platform Notes

| Platform | Status | Notes |
|----------|--------|-------|
| macOS (Apple Silicon) | ✅ Tested | Metal acceleration, fastest transcription |
| macOS (Intel) | ✅ Should work | No Metal, slower |
| Linux | 🔶 Untested | Needs `alsa`/`pulseaudio` for streaming; build whisper-cpp from source |
| Windows | ❌ Not supported | PRs welcome |

On Linux, install dependencies via your package manager instead of Homebrew, and build [whisper.cpp](https://github.com/ggerganov/whisper.cpp) from source.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/funkatron/infomux.git
cd infomux

# 2. Install system dependencies
brew install ffmpeg whisper-cpp

# 3. Download whisper model (~142 MB)
mkdir -p ~/.local/share/infomux/models/whisper
curl -L -o ~/.local/share/infomux/models/whisper/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# 4. Set model path (add to ~/.zshrc for persistence)
export INFOMUX_WHISPER_MODEL="$HOME/.local/share/infomux/models/whisper/ggml-base.en.bin"

# 5. Install infomux (using uv, or pip)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .

# 6. Verify everything works
infomux run --check-deps

# 7. Transcribe something!
infomux run your-podcast.mp4
```

> **Tip:** For summarization, also install [Ollama](https://ollama.ai) and run `ollama pull qwen2.5:7b-instruct`

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

### `infomux stream`

Real-time audio capture and transcription from a microphone.

```bash
# List available audio devices
infomux stream --list-devices

# Interactive device selection
infomux stream

# Use a specific device
infomux stream --device 3

# Stop after 60 seconds
infomux stream --duration 60

# Stop after 10 seconds of silence
infomux stream --silence 10

# Stop when phrase detected (default: "stop recording")
infomux stream --stop-word "end session"

# Combine options
infomux stream --device 3 --duration 120 --silence 10
```

**Stop conditions:**
- Press `Ctrl+C`
- Duration limit reached (`--duration`)
- Silence threshold exceeded (`--silence`)
- Stop phrase detected (`--stop-word`, default: "stop recording")

**Output artifacts:**
- `audio.wav` — The recorded audio
- `transcript.json` — Full JSON with word-level timestamps
- `transcript.srt` — SRT subtitles
- `transcript.vtt` — VTT subtitles

**Example session:**
```
──────────────────────────────────────────────────
  Recording from: M2

  Stop recording by:
    • Press Ctrl+C
    • Wait 60 seconds (auto-stop)
    • Say "stop recording"
──────────────────────────────────────────────────

[Start speaking]
 Hello, this is a test recording...
 Stop recording.

Stopping: stop word 'stop recording'
/Users/you/.local/share/infomux/runs/run-20260111-030000-abc123
```

---

## Pipelines

### Available Pipelines

| Pipeline | Steps | Requires |
|----------|-------|----------|
| `transcribe` (default) | extract_audio → transcribe | ffmpeg, whisper-cli |
| `summarize` | extract_audio → transcribe → summarize | ffmpeg, whisper-cli, Ollama |

```bash
# List available pipelines
infomux run --list-pipelines
```

### Steps

| Step | Input | Output | Tool |
|------|-------|--------|------|
| `extract_audio` | media file | `audio.wav` (16kHz mono) | ffmpeg |
| `transcribe` | `audio.wav` | `transcript.txt` | whisper-cli |
| `summarize` | `transcript.txt` | `summary.md` | Ollama |

### Data Flow

```
input.mp4 → [extract_audio] → audio.wav → [transcribe] → transcript.txt
                                                ↓
                                          [summarize] → summary.md
```

---

## Data Storage

### Run Directory

Each run creates a directory under `~/.local/share/infomux/runs/`:

```
~/.local/share/infomux/
├── runs/
│   ├── run-20260111-020549-c36c19/     # From 'infomux run'
│   │   ├── job.json          # Execution metadata
│   │   ├── audio.wav         # Extracted audio
│   │   └── transcript.txt    # Transcription
│   ├── run-20260111-030000-abc123/     # From 'infomux stream'
│   │   ├── job.json          # Execution metadata
│   │   ├── audio.wav         # Recorded audio
│   │   ├── transcript.json   # Full JSON with word-level timestamps
│   │   ├── transcript.srt    # SRT subtitles
│   │   └── transcript.vtt    # VTT subtitles
│   └── ...
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
| `INFOMUX_OLLAMA_MODEL` | Ollama model for summarization | `qwen2.5:7b-instruct` |
| `INFOMUX_OLLAMA_URL` | Ollama API URL | `http://localhost:11434` |

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

### `Ollama not running` (for summarization)

The `summarize` pipeline requires Ollama:

```bash
# Install Ollama
brew install ollama

# Start the server
ollama serve

# Pull a model (in another terminal)
ollama pull qwen2.5:7b-instruct
```

### `No audio devices found` (for streaming)

Ensure your microphone is connected and permissions are granted:

```bash
# List available devices
infomux stream --list-devices
```

On macOS, you may need to grant Terminal/your IDE microphone access in System Preferences → Privacy & Security → Microphone.

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
├── llm.py              # LLM reproducibility metadata (ModelInfo, GenerationParams)
├── audio.py            # Audio device discovery
├── pipeline.py         # Step orchestration
├── pipeline_def.py     # Pipeline definitions as data (PipelineDef, StepDef)
├── storage.py          # Run directory management
├── commands/
│   ├── run.py          # infomux run
│   ├── inspect.py      # infomux inspect
│   ├── resume.py       # infomux resume
│   └── stream.py       # infomux stream (real-time transcription)
└── steps/
    ├── __init__.py     # Step protocol and registry
    ├── extract_audio.py # ffmpeg wrapper
    ├── transcribe.py   # whisper-cli wrapper
    └── summarize.py    # Ollama LLM wrapper
```

---

## Implementation Status

### ✅ Implemented

- CLI scaffold with `run`, `inspect`, `resume`, `stream` subcommands
- Job envelope with input hashing, step timing, artifact tracking
- Run storage under `~/.local/share/infomux/runs/`
- `extract_audio` step (ffmpeg → 16kHz mono WAV)
- `transcribe` step (whisper-cli → transcript.txt)
- `summarize` step (Ollama → summary.md)
- Pipeline definitions as data (`PipelineDef`, `StepDef`)
- Step input/output dependency resolution
- `--pipeline` and `--list-pipelines` flags
- `--steps` flag for running specific steps
- `resume` command with `--from-step` support
- Dependency checking (`--check-deps`)
- Dry-run mode (`--dry-run`)
- Logging to stderr
- Model/seed recording for reproducibility
- `stream` command for real-time audio capture
- Word-level timestamps via whisper-cli post-processing
- Multiple stop conditions (duration, silence, stop-word)
- Audio device discovery and selection

### ❌ Planned

- **Frame extraction** — Key frames from video
- **Custom pipelines** — Load from YAML/JSON config file
- **Model auto-download** — `infomux setup` command
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
