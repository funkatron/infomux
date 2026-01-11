# infomux Architecture

This document explains the internal design of infomux for contributors and maintainers.

## Design Goals

1. **Auditability** — Every run is fully traceable via `job.json`
2. **Testability** — Each component is isolated and unit-testable
3. **Extensibility** — New steps can be added without modifying core code
4. **Simplicity** — Minimal dependencies, stdlib where possible

---

## Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           CLI Layer                             │
│  cli.py → commands/{run,inspect,resume}.py                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Layer                           │
│  pipeline.py — orchestrates step execution                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Steps Layer                             │
│  steps/{extract_audio,transcribe}.py — individual operations    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Layer                               │
│  job.py — data structures                                       │
│  storage.py — file I/O                                          │
│  config.py — tool discovery                                     │
│  log.py — logging                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Data Structures

### `JobEnvelope` (job.py)

The central record of a pipeline run. Serialized to `job.json`.

```python
@dataclass
class JobEnvelope:
    id: str                      # run-YYYYMMDD-HHMMSS-XXXXXX
    created_at: str              # ISO 8601 timestamp
    updated_at: str              # ISO 8601 timestamp
    status: str                  # pending|running|completed|failed|interrupted
    input: InputFile | None      # Input file metadata
    steps: list[StepRecord]      # Execution records per step
    artifacts: list[str]         # Output file paths
    config: dict[str, Any]       # User configuration
    error: str | None            # Error message if failed
```

### `InputFile` (job.py)

Captures input file metadata for verification and reproducibility.

```python
@dataclass
class InputFile:
    path: str           # Absolute path
    sha256: str         # Content hash for verification
    size_bytes: int     # File size
```

### `StepRecord` (job.py)

Records execution of a single pipeline step.

```python
@dataclass
class StepRecord:
    name: str                       # Step name
    status: str                     # pending|running|completed|failed
    started_at: str | None          # ISO 8601
    completed_at: str | None        # ISO 8601
    duration_seconds: float | None  # Elapsed time
    error: str | None               # Error if failed
    outputs: list[str]              # Artifact paths
```

### `StepResult` (steps/__init__.py)

Returned by step execution.

```python
@dataclass
class StepResult:
    name: str
    success: bool
    outputs: list[Path]
    duration_seconds: float
    error: str | None = None
```

---

## Execution Flow

### `infomux run input.mp4`

```
1. CLI parsing (cli.py)
   └── Parse args, configure logging

2. Input validation (commands/run.py)
   ├── Check file exists
   └── Compute SHA-256 hash

3. Job creation (job.py)
   ├── Generate run ID
   ├── Create JobEnvelope
   └── Save to ~/.local/share/infomux/runs/<run-id>/job.json

4. Pipeline execution (pipeline.py)
   └── For each step:
       ├── Create StepRecord (status=running)
       ├── Execute step (steps/*.py)
       ├── Update StepRecord with results
       └── Save job.json

5. Output (commands/run.py)
   ├── Print run directory to stdout
   └── Exit with 0 (success) or 1 (failure)
```

---

## Adding a New Step

### 1. Create step module

```python
# src/infomux/steps/summarize.py

from dataclasses import dataclass
from pathlib import Path

from infomux.steps import StepError, StepResult, register_step

@register_step
@dataclass
class SummarizeStep:
    name: str = "summarize"

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        # Read transcript
        transcript = input_path.read_text()
        
        # Generate summary (placeholder)
        summary = f"Summary of {len(transcript)} characters..."
        
        # Write output
        output_path = output_dir / "summary.md"
        output_path.write_text(summary)
        
        return [output_path]
```

### 2. Register in pipeline

```python
# src/infomux/pipeline.py

DEFAULT_STEPS = ["extract_audio", "transcribe", "summarize"]

def _run_step(step_name: str, input_path: Path, output_dir: Path) -> StepResult:
    # ...existing cases...
    elif step_name == "summarize":
        return run_summarize(input_path, output_dir)
```

### 3. Add tests

```python
# tests/test_steps.py

def test_summarize_step(tmp_path):
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("Hello world")
    
    step = SummarizeStep()
    outputs = step.execute(transcript, tmp_path)
    
    assert len(outputs) == 1
    assert outputs[0].name == "summary.md"
    assert outputs[0].exists()
```

---

## I/O Conventions

### stdout vs stderr

| Stream | Content | Example |
|--------|---------|---------|
| stdout | Machine-readable output | Run directory path, JSON |
| stderr | Human-readable logs | Progress, errors, debug info |

This enables piping:

```bash
RUN_DIR=$(infomux run input.mp4)
cat "$RUN_DIR/transcript.txt"
```

### File naming

| File | Purpose | Format |
|------|---------|--------|
| `job.json` | Execution metadata | JSON |
| `audio.wav` | Extracted audio | 16kHz mono PCM WAV |
| `transcript.txt` | Transcription | Plain text |

---

## Error Handling

### StepError

Steps raise `StepError` for recoverable failures:

```python
class StepError(Exception):
    def __init__(self, step_name: str, message: str):
        self.step_name = step_name
        self.message = message
```

### Pipeline behavior

1. Step raises `StepError`
2. Pipeline catches and records in `StepRecord.error`
3. Job status set to `failed`
4. `job.json` saved with error details
5. CLI exits with code 1

---

## Testing Strategy

### Unit tests

- `test_job.py` — JobEnvelope serialization
- `test_cli.py` — Argument parsing

### Integration tests (TODO)

- Full pipeline execution with sample files
- Resume from failed state

### Test fixtures

Use `pytest`'s `tmp_path` fixture for isolated test directories:

```python
def test_example(tmp_path: Path):
    test_file = tmp_path / "input.txt"
    test_file.write_text("content")
    # ...
```

---

## Future Considerations

### Custom pipelines

Allow users to define step sequences:

```yaml
# infomux.yaml
pipeline:
  - extract_audio
  - transcribe
  - summarize:
      model: qwen2.5:14b-instruct
```

### Parallel execution

Some steps could run in parallel:

```
input.mp4
    ├── [extract_audio] → audio.wav → [transcribe] → transcript.txt
    └── [extract_frames] → frames/*.jpg
```

### Remote execution

Optional remote processing with explicit configuration:

```bash
INFOMUX_REMOTE_URL=https://api.example.com infomux run input.mp4
```

---

## Code Style

- Python 3.11+ features (type hints, dataclasses)
- `ruff` for linting and formatting
- Docstrings on public functions
- Explicit over implicit
- No magic, no metaprogramming
