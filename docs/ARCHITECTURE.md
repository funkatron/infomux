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
│  cli.py → commands/{run,inspect,resume,stream}.py               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Layer                           │
│  pipeline.py — orchestrates step execution                      │
│  pipeline_def.py — declarative pipeline definitions             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Steps Layer                             │
│  steps/__init__.py — auto-discovery, protocol, registry         │
│  steps/*.py — individual step implementations                   │
│  (13 steps: extract, transcribe, summarize, embed, stores...)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Layer                               │
│  job.py — JobEnvelope, StepRecord dataclasses                   │
│  llm.py — ModelInfo, GenerationParams (reproducibility)         │
│  storage.py — run directory management                          │
│  config.py — tool/model discovery                               │
│  audio.py — device enumeration                                  │
│  log.py — stderr logging                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Data Structures

### `PipelineDef` (pipeline_def.py)

Defines a pipeline as data, enabling declarative configuration:

```python
@dataclass
class StepDef:
    name: str                       # Step identifier
    input_from: str | None = None   # Which step's output to use (None = original input)
    config: dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineDef:
    name: str                       # Pipeline name
    description: str                # What this pipeline does
    steps: list[StepDef]            # Ordered list of steps
```

The default pipeline is defined as:

```python
DEFAULT_PIPELINE = PipelineDef(
    name="transcribe",
    description="Extract audio and transcribe to text",
    steps=[
        StepDef(name="extract_audio", input_from=None),
        StepDef(name="transcribe", input_from="extract_audio"),
    ],
)
```

This structure enables:
- Loading pipelines from YAML/JSON config files (future)
- Validating step dependencies
- Serializing pipeline definitions in job envelopes

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

Steps are auto-discovered from the `steps/` directory. Just create a module with a `@register_step` decorated class.

### 1. Create step module

```python
# src/infomux/steps/my_step.py

from dataclasses import dataclass
from pathlib import Path

from infomux.steps import StepError, register_step

@register_step
@dataclass
class MyStep:
    """
    Docstring explaining what this step does.
    """
    name: str = "my_step"  # Must match the expected step name

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Execute the step.

        Args:
            input_path: Path to input file (from previous step or original input)
            output_dir: Run directory to write outputs

        Returns:
            List of output file paths

        Raises:
            StepError: If step fails
        """
        # Read input
        content = input_path.read_text()

        # Process (your logic here)
        result = f"Processed: {len(content)} chars"

        # Write output
        output_path = output_dir / "my_output.txt"
        output_path.write_text(result)

        return [output_path]
```

### 2. Define output in step registry

The step registry maps step names to their primary output files:

```python
# src/infomux/steps/__init__.py

STEP_OUTPUTS = {
    # ... existing steps ...
    "my_step": "my_output.txt",
}
```

### 3. Add to a pipeline

```python
# src/infomux/pipeline_def.py

MY_PIPELINE = PipelineDef(
    name="my_pipeline",
    description="Does my custom processing",
    steps=[
        StepDef(name="extract_audio", input_from=None),
        StepDef(name="transcribe", input_from="extract_audio"),
        StepDef(name="my_step", input_from="transcribe"),
    ],
)

# Register in PIPELINES dict
PIPELINES["my_pipeline"] = MY_PIPELINE
```

### 4. Add tests

```python
# tests/test_my_step.py

import pytest
from pathlib import Path
from infomux.steps.my_step import MyStep

class TestMyStep:
    def test_produces_output(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.txt"
        input_file.write_text("Hello world")

        step = MyStep()
        outputs = step.execute(input_file, tmp_path)

        assert len(outputs) == 1
        assert outputs[0].name == "my_output.txt"
        assert outputs[0].exists()

    def test_handles_empty_input(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.txt"
        input_file.write_text("")

        step = MyStep()
        # Should raise StepError or handle gracefully
        with pytest.raises(Exception):
            step.execute(input_file, tmp_path)
```

### Step conventions

- **Always** use `@register_step` decorator
- **Always** use `@dataclass` decorator
- Return `list[Path]` of output files
- Raise `StepError(self.name, "message")` on failures
- Log progress with `logger.info()`, details with `logger.debug()`
- For LLM steps, return `tuple[list[Path], StepModelRecord]` for reproducibility

---

## LLM Integration (Summarization)

The `summarize` step uses Ollama for local LLM inference with reproducibility tracking.

### Chunked Processing

Long transcripts (>15k chars) are automatically chunked:

```
┌─────────────┐
│ Transcript  │
│ (66k chars) │
└─────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│         _chunk_text()                     │
│  Split at sentence boundaries (~12k each) │
└──────────────────────────────────────────┘
       │
       ▼
┌─────────┬─────────┬─────────┬─────────┐
│ Chunk 1 │ Chunk 2 │ Chunk 3 │ Chunk 4 │
└────┬────┴────┬────┴────┬────┴────┬────┘
     │         │         │         │
     ▼         ▼         ▼         ▼
┌────────────────────────────────────────┐
│     LLM Extract (per chunk)             │
│  Action items, decisions, quotes, etc.  │
└────────────────────────────────────────┘
     │         │         │         │
     └────────┬┴─────────┴┬────────┘
              │           │
              ▼           │
┌────────────────────────────────────────┐
│        LLM Combine                      │
│  Merge all extracts into final summary  │
└────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│              summary.md                  │
│  ## Overview                             │
│  ## Action Items                         │
│  ## Key Takeaways                        │
│  ...                                     │
└─────────────────────────────────────────┘
```

### Content Type Adaptation

The `INFOMUX_CONTENT_TYPE_HINT` environment variable (or `--content-type-hint` CLI flag) adapts the prompt:

```python
CONTENT_TYPE_HINTS = {
    "meeting": "Focus on: action items, decisions, assignments, deadlines.",
    "talk": "Focus on: key concepts, takeaways, quotes.",
    "podcast": "Focus on: main topics, guest insights, recommendations.",
    # ...
}
```

Custom strings are passed directly to the model.

### Reproducibility

LLM steps return `StepModelRecord` with:

```python
@dataclass
class StepModelRecord:
    model: ModelInfo           # name, provider, version, path
    params: GenerationParams   # seed, temperature, top_p, max_tokens
    input_hash: str            # SHA-256 of input text
    output_tokens: int         # Tokens generated
```

This enables:
- Reproducing exact outputs with same seed
- Tracking which model versions produced which outputs
- Comparing results across model changes

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
