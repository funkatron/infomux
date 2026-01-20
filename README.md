# infomux

A local-first CLI for transcribing audio/video and capturing voice notes.

**What it does:**
- Transcribe any audio/video file to text
- Record voice notes from your microphone with live transcription
- Generate summaries using local LLMs (Ollama)
- Keep everything on your machine — no cloud, no API keys

```bash
# Transcribe a podcast episode
infomux run ~/Downloads/episode-42.mp3
# → ~/.local/share/infomux/runs/run-XXXXXX/transcript.txt

# Record a voice memo with timestamps
infomux stream --duration 300
# → audio.wav + transcript.srt/vtt/json

# Get summary of a meeting recording
infomux run --pipeline summarize zoom-call.mp4
# → transcript.txt + summary.md

# Add subtitles to a music video
infomux run --pipeline caption my-song.mp4
# → video with embedded toggleable subtitles

# Generate video from audio with burned subtitles
infomux run --pipeline audio-to-video voice-note.m4a
# → video with burned-in subtitles (great for sharing!)

# Generate lyric video with word-level burned subtitles
infomux run --pipeline lyric-video song.mp3
# → video with each word appearing at its exact timing

# Customize lyric video appearance
infomux run --pipeline lyric-video --lyric-font-size 60 --lyric-font-color yellow --lyric-position top song.mp3

# Full analysis: transcript + timestamps + summary + database
infomux run --pipeline report-store interview.m4a
# → all outputs + indexed in searchable SQLite
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
| macOS (Intel) | 🤷‍♀️ Should work | No Metal, slower |
| Linux | 🔶 Untested | See known issues below |
| Windows | ❌ Not supported | PRs welcome |

**Linux known/probable issues:**

1. **Audio device discovery** — Uses `ffmpeg -f avfoundation` which is macOS-only. Linux needs `-f alsa` or `-f pulse`. The `audio.py` module would need platform detection.

2. **whisper-cpp** — Not in most package managers. Build from [source](https://github.com/ggerganov/whisper.cpp) or use a PPA/AUR package.

3. **whisper-stream** — May need different audio backend flags for ALSA/PulseAudio.

Core functionality (`infomux run` for file transcription) should work if whisper-cli and ffmpeg are installed.

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
uv sync && uv pip install -e .

# 6. Verify everything works
infomux run --check-deps

# 7. Transcribe something!
infomux run your-podcast.mp4
```

> **Tip:** For summarization, install [Ollama](https://ollama.ai) and pull a model:
> ```bash
> ollama pull llama3.1:8b          # Default, 8GB RAM
> ollama pull qwen2.5:32b-instruct  # Better quality, 20GB RAM
> ```

---

## Supported Input Formats

infomux accepts any audio or video format that ffmpeg can decode. The `extract_audio` step automatically converts to 16kHz mono WAV for whisper.

### Video
| Format | Extension | Notes |
|--------|-----------|-------|
| MP4 | `.mp4` | Most common, recommended |
| QuickTime | `.mov` | Native macOS format |
| Matroska | `.mkv` | Common for downloads |
| WebM | `.webm` | YouTube/web downloads |
| AVI | `.avi` | Legacy Windows format |

### Audio
| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | Uncompressed, best quality |
| MP3 | `.mp3` | Most common compressed |
| FLAC | `.flac` | Lossless compressed |
| AAC/M4A | `.m4a`, `.aac` | Apple/podcast format |
| Ogg Vorbis | `.ogg` | Open format |

### Not Supported
- **Images** — no audio to transcribe
- **Streams** — use `infomux stream` for live capture
- **Encrypted files** — DRM-protected content won't decode

> **Tip:** If ffmpeg can play it, infomux can process it. Test with: `ffmpeg -i yourfile.xyz`

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
# Transcribe an audio file (uses default 'transcribe' pipeline)
infomux run ~/Music/interview.m4a

# Transcribe a video, extract audio automatically
infomux run ~/Movies/lecture.mp4

# Get a summary of a long recording
infomux run --pipeline summarize 3hr-meeting.mp4

# Summarize with smarter model and content hint
infomux run --pipeline summarize --model qwen2.5:32b-instruct --content-type-hint meeting standup.mp4

# Summarize a conference talk (adapts output for key takeaways)
infomux run --pipeline summarize --content-type-hint talk keynote.mp4

# Create subtitles for a video (soft subs, toggleable)
infomux run --pipeline caption my-music-video.mp4

# Burn subtitles into video permanently
infomux run --pipeline caption-burn tutorial.mp4

# Get word-level timestamps without video
infomux run --pipeline timed podcast.mp3

# Generate video from audio with burned subtitles
infomux run --pipeline audio-to-video meeting-recording.m4a

# Customize video background and size
infomux run --pipeline audio-to-video --video-background-color blue --video-size 1280x720 audio.m4a

# Use custom background image
infomux run --pipeline audio-to-video --video-background-image ~/Pictures/bg.png audio.m4a

# Generate lyric video with word-level burned subtitles
infomux run --pipeline lyric-video song.mp3

# Customize lyric video appearance
infomux run --pipeline lyric-video --lyric-font-size 60 --lyric-font-color yellow --lyric-position top song.mp3
infomux run --pipeline lyric-video --lyric-font-name "Helvetica" --lyric-word-spacing 30 song.mp3

# Full analysis with searchable database
infomux run --pipeline report-store weekly-standup.mp4

# List all available pipelines (use inspect command)
infomux inspect --list-pipelines

# List all available steps (use inspect command)
infomux inspect --list-steps

# Preview what would happen (no actual processing)
infomux run --dry-run my-file.mp4

# Check that ffmpeg, whisper-cli, and model are installed
infomux run --check-deps

# Verbose logging (shows debug output)
infomux -v run my-file.mp4
```

**Output:** Prints the run directory path to stdout.

### `infomux inspect`

View details of a completed run.

```bash
# List all runs with summary information (tabular format)
infomux inspect --list

# List runs as JSON (for scripting/automation)
infomux inspect --list --json

# List available pipelines
infomux inspect --list-pipelines

# List available steps
infomux inspect --list-steps

# View a specific run (tab-complete the run ID)
infomux inspect run-20260111-020549-c36c19

# Show the path to a run directory
infomux inspect --path run-20260111-020549-c36c19

# Open the run directory in Finder (macOS) or file manager
infomux inspect --open run-20260111-020549-c36c19

# Get JSON for scripting/automation
infomux inspect --json run-20260111-020549-c36c19

# Pipe to jq for specific fields
infomux inspect --json run-XXXXX | jq '.artifacts'
```

**Example output (inspect --list):**

```
   Run ID                     Status    Date       Pipeline       Input                                          Artifacts
--------------------------------------------------------------------------------------------------------------------------
●  run-20260120-191406-7179cd completed 2026-01-20 caption-burn   audio.simplecast.com....mp3                            5
●  run-20260120-190809-ae6458 completed 2026-01-20 transcribe     audio.simplecast.com....mp3                            2
●  run-20260113-003525-1a50f0 completed 2026-01-13 timed          Skin WBD NEO team mtg 2025-06-25.m4a                   6
●  run-20260113-002820-c4ae2c completed 2026-01-13 audio-to-video how_to_be_a_great_developer_tek14-lossless.m4a         7

Total: 4 run(s)
```

**Example output (inspect <run-id>):**

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
# Resume a failed/interrupted run
infomux resume run-20260111-020549-c36c19

# Re-run transcription (e.g., after updating whisper model)
infomux resume --from-step transcribe run-XXXXX

# Re-generate summary with different Ollama model
infomux resume --from-step summarize --model qwen2.5:32b-instruct run-XXXXX

# Re-summarize with content type hint (adapts output format)
infomux resume --from-step summarize --content-type-hint meeting run-XXXXX
infomux resume --from-step summarize --content-type-hint talk run-XXXXX

# Preview what would be re-run
infomux resume --dry-run run-XXXXX
```

**Behavior:**
- Loads existing job envelope from the run directory
- Skips already-completed steps (unless `--from-step` specified)
- Clears failed step records before re-running
- Uses the same pipeline and input as the original run

### `infomux cleanup`

Remove orphaned or unwanted runs from the runs directory.

```bash
# Preview what would be deleted (always use this first!)
infomux cleanup --dry-run --orphaned

# Delete runs without valid job.json files
infomux cleanup --force --orphaned

# Delete stuck runs (status: running)
infomux cleanup --force --status running

# Delete runs older than 30 days
infomux cleanup --force --older-than 30d

# Delete failed runs older than 7 days (safety check)
infomux cleanup --force --status failed --older-than 7d --min-age 1d

# Combine filters: delete orphaned runs and stuck runs
infomux cleanup --force --orphaned --status running
```

**Filters:**
- `--orphaned`: Delete runs without valid `job.json` files
- `--status <status>`: Delete runs with specific status (`pending`, `running`, `failed`, `interrupted`, `completed`)
- `--older-than <time>`: Delete runs older than specified time (e.g., `30d`, `2w`, `1m`)

**Safety:**
- Always use `--dry-run` first to preview what would be deleted
- `--force` is required to actually delete (prevents accidental deletion)
- `--min-age` can be used as a safety check to prevent deleting very recent runs

**Time specifications:**
- `d` = days (e.g., `30d` = 30 days)
- `w` = weeks (e.g., `2w` = 2 weeks)
- `m` = months (e.g., `1m` = 30 days)

**Example output:**
```
Would delete 4 run(s):

  run-20260111-025200-449ae0 (status: running)
  run-20260111-025752-b546d0 (status: running)
  run-20260111-025832-99d059 (status: running)
  run-20260113-002114-f80d18 (status: running)

Run with --force to actually delete these runs.
```

### `infomux stream`

Real-time audio capture and transcription from a microphone.

```bash
# See available microphones
infomux stream --list-devices

# Record with interactive device picker
infomux stream

# Use a specific microphone (by ID from --list-devices)
infomux stream --device 2

# 5-minute voice memo
infomux stream --duration 300

# Auto-stop after 5 seconds of silence (great for dictation)
infomux stream --silence 5

# Custom stop phrase
infomux stream --stop-word "end note"

# Voice memo with summarization
infomux stream --pipeline summarize

# Meeting notes with auto-silence detection
infomux stream --device 2 --silence 10 --pipeline summarize

# Show available pipelines for stream
infomux stream --list-pipelines
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

| Pipeline | Description | Steps |
|----------|-------------|-------|
| `transcribe` | Plain text transcript (default) | extract_audio → transcribe |
| `summarize` | Transcript + LLM summary | extract_audio → transcribe → summarize |
| `timed` | Word-level timestamps (SRT/VTT/JSON) | extract_audio → transcribe_timed |
| `report` | Full analysis: text, timestamps, summary | ... → transcribe → transcribe_timed → summarize |
| `report-store` | Full analysis + searchable database | ... → summarize → store_sqlite |
| `caption` | Soft subtitles (toggleable) | extract_audio → transcribe_timed → embed_subs |
| `caption-burn` | Burned-in subtitles (permanent) | extract_audio → transcribe_timed → embed_subs |
| `audio-to-video` | Generate video from audio with burned subtitles | extract_audio → transcribe_timed → generate_video |
| `lyric-video` | Generate lyric video with word-level burned subtitles | extract_audio → transcribe_timed → generate_lyric_video |
| `lyric-video-isolated` | Generate lyric video with vocal isolation for improved timing | extract_audio → isolate_vocals → transcribe_timed → generate_lyric_video |

```bash
# List available pipelines
infomux inspect --list-pipelines

# List available steps
infomux inspect --list-steps
```

### Steps

| Step | Input | Output | Tool |
|------|-------|--------|------|
| `extract_audio` | media file | `audio.wav` (16kHz mono) | ffmpeg |
| `isolate_vocals` | `audio.wav` | `audio_vocals.wav` (isolated vocals) | demucs or spleeter |
| `transcribe` | `audio.wav` | `transcript.txt` | whisper-cli |
| `transcribe_timed` | `audio.wav` | `transcript.srt`, `.vtt`, `.json` | whisper-cli -dtw |
| `summarize` | `transcript.txt` | `summary.md` | Ollama (chunked for long input) |
| `embed_subs` | video + `.srt` | `video_captioned.mp4` | ffmpeg |
| `generate_video` | audio + `.srt` | `audio_with_subs.mp4` | ffmpeg |
| `generate_lyric_video` | audio + `transcript.json` | `audio_lyric_video.mp4` | ffmpeg |
| `store_json` | run directory | `report.json` | (built-in) |
| `store_markdown` | run directory | `report.md` | (built-in) |
| `store_sqlite` | run directory | → `infomux.db` | sqlite3 |
| `store_s3` | run directory | → S3 bucket | boto3 |
| `store_postgres` | run directory | → PostgreSQL | psycopg2 |
| `store_obsidian` | run directory | → Obsidian vault | (built-in) |
| `store_bear` | run directory | → Bear.app | macOS only |

### Data Flow

```
# transcribe pipeline (default)
input.mp4 → [extract_audio] → audio.wav → [transcribe] → transcript.txt

# summarize pipeline
input.mp4 → [extract_audio] → audio.wav → [transcribe] → transcript.txt
                                                 ↓
                                           [summarize] → summary.md

# caption pipeline (for music videos, lyrics)
input.mp4 → [extract_audio] → audio.wav → [transcribe_timed] → transcript.srt/vtt/json
    ↓                                                                    ↓
    └───────────────────────────────────→ [embed_subs] ←─────────────────┘
                                               ↓
                                    video_captioned.mp4 (with soft subtitles)

# audio-to-video pipeline (generate video from audio)
input.m4a → [extract_audio] → audio.wav → [transcribe_timed] → transcript.srt/vtt/json
                                                                    ↓
                                                          [generate_video] → audio_with_subs.mp4
                                                          (solid color or image background)

# lyric-video pipeline (word-level lyric video)
input.m4a → [extract_audio] → audio.wav → [transcribe_timed] → transcript.json (word-level)
                                                                    ↓
                                                          [generate_lyric_video] → audio_lyric_video.mp4
                                                          (each word appears at exact timing)

# lyric-video-isolated pipeline (with vocal isolation for better timing)
input.m4a → [extract_audio] → audio.wav → [isolate_vocals] → audio_vocals.wav → [transcribe_timed] → transcript.json
                                                                                                        ↓
                                                                                          [generate_lyric_video] → audio_lyric_video.mp4
                                                                                          (uses original audio.wav for video, isolated vocals for timing)
```

### Pipeline Artifacts

Each pipeline produces different output files:

**`transcribe`** (default)
```
├── audio.wav          # 16kHz mono audio
├── transcript.txt     # Plain text transcript
└── job.json
```

**`timed`**
```
├── audio.wav
├── transcript.srt     # SRT subtitles
├── transcript.vtt     # VTT subtitles
├── transcript.json    # Word-level timestamps
└── job.json
```

**`summarize`**
```
├── audio.wav
├── transcript.txt
├── summary.md         # LLM-generated summary
└── job.json
```

**`report`** (full analysis)
```
├── audio.wav
├── transcript.txt     # Plain text
├── transcript.srt     # SRT subtitles
├── transcript.vtt     # VTT subtitles
├── transcript.json    # Word-level timestamps
├── summary.md         # LLM summary
└── job.json
```

**`report-store`** (full analysis + database)
```
├── (same as report)
└── → ~/.local/share/infomux/infomux.db  # Searchable database
```

The SQLite database enables:
- Full-text search across all transcripts
- Segment-level queries with timestamps
- Summary aggregation across runs

**`caption`** / **`caption-burn`**
```
├── audio.wav
├── transcript.srt
├── transcript.vtt
├── transcript.json
├── video_captioned.mp4  # Video with subtitles
└── job.json
```

**`audio-to-video`**
```
├── audio.wav
├── transcript.srt
├── transcript.vtt
├── transcript.json
├── audio_with_subs.mp4  # Generated video with burned subtitles
└── job.json
```

> **Note:** The `audio-to-video` pipeline generates a video file from audio with a solid color or image background. Use `--video-background-image`, `--video-background-color`, or `--video-size` to customize the output.

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
| `INFOMUX_OLLAMA_MODEL` | Ollama model for summarization | `llama3.1:8b` |
| `INFOMUX_OLLAMA_URL` | Ollama API URL | `http://localhost:11434` |
| `INFOMUX_CONTENT_TYPE_HINT` | Hint for content type (meeting, talk, etc.) | *(none)* |
| `INFOMUX_S3_BUCKET` | S3 bucket for `store_s3` | *(required if using S3)* |
| `INFOMUX_S3_PREFIX` | S3 key prefix | `infomux/` |
| `INFOMUX_POSTGRES_URL` | PostgreSQL connection URL for `store_postgres` | *(required if using PG)* |
| `INFOMUX_OBSIDIAN_VAULT` | Path to Obsidian vault for `store_obsidian` | *(required if using Obsidian)* |
| `INFOMUX_OBSIDIAN_FOLDER` | Subfolder in vault for transcripts | `Transcripts` |
| `INFOMUX_OBSIDIAN_TAGS` | Comma-separated default tags | `infomux,transcript` |
| `INFOMUX_BEAR_TAGS` | Comma-separated default tags for Bear | `infomux,transcript` |

### Summarization Options

The `summarize` step uses Ollama for local LLM inference. For best results:

```bash
# Recommended: pull a 32B model for better accuracy (requires ~20GB VRAM/RAM)
ollama pull qwen2.5:32b-instruct

# Use it via CLI flag
infomux run --pipeline summarize --model qwen2.5:32b-instruct meeting.mp4
```

**Content Type Hints**

Adapt summarization output for different content types:

| Hint | Focus | Best for |
|------|-------|----------|
| `meeting` | Action items, decisions, deadlines | Work meetings, standups |
| `talk` | Key concepts, takeaways, quotes | Conference talks, presentations |
| `podcast` | Main topics, guest insights | Interviews, podcasts |
| `lecture` | Concepts, examples, definitions | Educational content |
| `standup` | Blockers, progress, next steps | Daily standups |
| `1on1` | Feedback, goals, concerns | One-on-one meetings |

Or pass any custom string:

```bash
infomux run --pipeline summarize --content-type-hint "quarterly review" recording.mp4
```

**Long Transcript Handling**

Transcripts over 15,000 characters are automatically chunked and processed sequentially to ensure full coverage. You'll see progress like:

```
chunk 1/4 (0%)
chunk 2/4 (25%), ~73s remaining
...
summarization complete: 139.9s total (combine: 42.5s)
```

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

### `demucs not found` or `spleeter not found` (for vocal isolation)

The `isolate_vocals` step requires either Demucs or Spleeter:

```bash
# Install Demucs (recommended for better quality)
uv pip install demucs

# Or install Spleeter (faster but lower quality)
uv pip install spleeter
```

Then use the `lyric-video-vocals` pipeline:

```bash
uv run infomux run --pipeline lyric-video-vocals <your-audio-file>
```

### `aeneas not found` (for forced alignment)

The `align_lyrics` step requires aeneas for forced alignment when you have official lyrics:

```bash
# Install system dependency (macOS)
brew install espeak

# Install Python dependencies (numpy must be installed first)
uv pip install numpy
uv pip install aeneas
```

**Note:** aeneas requires numpy to be installed before building. Install them separately as shown above.

Then use the `lyric-video-aligned` or `lyric-video-aligned-vocals` pipeline with a lyrics file:

```bash
uv run infomux run --pipeline lyric-video-aligned-vocals --lyrics-file lyrics.txt <your-audio-file>
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
    ├── __init__.py        # Step protocol, registry, auto-discovery
    ├── extract_audio.py   # ffmpeg wrapper
    ├── transcribe.py      # whisper-cli → transcript.txt
    ├── transcribe_timed.py # whisper-cli → .srt/.vtt/.json
    ├── summarize.py       # Ollama LLM (with chunking)
    ├── embed_subs.py      # ffmpeg subtitle embedding
    ├── storage.py         # Common storage API
    ├── store_json.py      # Export to JSON
    ├── store_markdown.py  # Export to Markdown
    ├── store_sqlite.py    # Index to SQLite
    ├── store_s3.py        # Upload to S3
    ├── store_postgres.py  # Index to PostgreSQL
    ├── store_obsidian.py  # Export to Obsidian vault
    └── store_bear.py      # Export to Bear.app (macOS)
```

---

## Implementation Status

### ✅ Implemented

**Core:**
- CLI with `run`, `inspect`, `resume`, `stream` subcommands
- Job envelope with input hashing, step timing, artifact tracking
- Run storage under `~/.local/share/infomux/runs/`
- Pipeline definitions as data (`PipelineDef`, `StepDef`)
- Auto-discovery of steps from `steps/` directory
- `--pipeline`, `--steps`, `--dry-run`, `--check-deps` flags (listing moved to `inspect` command)

**Steps:**
- `extract_audio` — ffmpeg → 16kHz mono WAV
- `isolate_vocals` — demucs/spleeter → isolated vocal track (optional, improves timing)
- `transcribe` — whisper-cli → transcript.txt
- `transcribe_timed` — whisper-cli -dtw → .srt/.vtt/.json
- `summarize` — Ollama with chunking, content hints, `--model` override
- `embed_subs` — ffmpeg subtitle embedding (soft or burned)
- `store_json`, `store_markdown` — export formats
- `store_sqlite` — searchable FTS5 database
- `store_s3`, `store_postgres` — cloud storage
- `store_obsidian`, `store_bear` — note app integration

**Pipelines:**
- `transcribe`, `summarize`, `timed`, `report`, `report-store`
- `caption`, `caption-burn` — video subtitle embedding
- `lyric-video`, `lyric-video-vocals` — word-level lyric videos
- `lyric-video-aligned`, `lyric-video-aligned-vocals` — forced alignment (requires aeneas)

**Streaming:**
- Real-time audio capture and transcription
- Multiple stop conditions (duration, silence, stop-word)
- Audio device discovery and selection

**Reproducibility:**
- Model/seed recording for LLM outputs
- Input file hashing (SHA-256)
- Full execution trace in job.json

### ❌ Planned

- **Frame extraction** — Key frames from video
- **Custom pipelines** — Load from YAML/JSON config file
- **Model auto-download** — `infomux setup` command
- **Parallel chunk processing** — Speed up long transcript summarization

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
