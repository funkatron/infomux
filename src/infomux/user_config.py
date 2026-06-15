"""
User-editable configuration for infomux (TOML).

Loads ~/.config/infomux/config.toml (or INFOMUX_CONFIG) for durable settings:
defaults, [[watch]] entries, and future pipeline definitions.

Precedence for pipeline options: CLI flags > config.toml [defaults] / [[watch]] > built-ins.
Secrets stay in .env (INFOMUX_*), not in this file.
"""

from __future__ import annotations

import os
import tomllib
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infomux.commands.run import RUN_PIPELINE_FIELDS
from infomux.log import get_logger

logger = get_logger(__name__)

ENV_CONFIG_PATH = "INFOMUX_CONFIG"

PATH_PIPELINE_FIELDS = frozenset(
    {
        "video_background_image",
        "lyric_font_file",
        "lyrics_file",
        "lyric_background_image",
    }
)

BOOL_PIPELINE_FIELDS = frozenset({"dry_run", "word_level_subtitles"})


def default_config_path() -> Path:
    """Return the default user config file path."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        base = Path(xdg).expanduser()
    else:
        base = Path.home() / ".config"
    return base / "infomux" / "config.toml"


def resolve_config_path(explicit: Path | str | None = None) -> Path:
    """Resolve config path from explicit arg, env, or default."""
    if explicit is not None:
        return Path(explicit).expanduser()
    env_path = os.environ.get(ENV_CONFIG_PATH)
    if env_path:
        return Path(env_path).expanduser()
    return default_config_path()


@dataclass
class PipelineOptions:
    """Pipeline flags shared by run and watch."""

    pipeline: str | None = None
    steps: str | None = None
    dry_run: bool = False
    model: str | None = None
    openai_model: str | None = None
    openai_base_url: str | None = None
    content_type_hint: str | None = None
    word_level_subtitles: bool = False
    video_background_image: Path | None = None
    video_background_color: str | None = None
    video_size: str | None = None
    lyric_font_name: str | None = None
    lyric_font_file: Path | None = None
    lyric_font_size: int | None = None
    lyrics_file: Path | None = None
    alignment_model: str | None = None
    lyric_font_color: str | None = None
    lyric_position: str | None = None
    lyric_word_spacing: int | None = None
    lyric_background_gradient: str | None = None
    lyric_background_image: Path | None = None
    _explicit: frozenset[str] = field(default_factory=frozenset, repr=False)

    def merged_with(self, overrides: PipelineOptions) -> PipelineOptions:
        """Return a copy with overrides applied (respecting explicit boolean false)."""
        data = {name: getattr(self, name) for name in RUN_PIPELINE_FIELDS}
        for name in RUN_PIPELINE_FIELDS:
            if name in overrides._explicit:
                data[name] = getattr(overrides, name)
            elif name not in BOOL_PIPELINE_FIELDS:
                override = getattr(overrides, name)
                if override is not None:
                    data[name] = override
        return PipelineOptions(
            **data,
            _explicit=self._explicit | overrides._explicit,
        )

    def to_namespace(self) -> Namespace:
        """Build an argparse Namespace for build_run_namespace()."""
        return Namespace(**{name: getattr(self, name) for name in RUN_PIPELINE_FIELDS})


@dataclass
class WatchEntry:
    """One [[watch]] table from config.toml."""

    directory: Path
    glob: str = "*"
    debounce: float = 2.0
    recursive: bool = True
    registry: Path | None = None
    pipeline: PipelineOptions = field(default_factory=PipelineOptions)


@dataclass
class ServeOptions:
    """Optional [serve] section for watch serve."""

    fswatch: Path | None = None


@dataclass
class UserConfig:
    """Parsed infomux user configuration."""

    path: Path | None
    defaults: PipelineOptions = field(default_factory=PipelineOptions)
    watches: list[WatchEntry] = field(default_factory=list)
    serve: ServeOptions = field(default_factory=ServeOptions)


def _coerce_pipeline_value(name: str, value: Any) -> Any:
    if name in PATH_PIPELINE_FIELDS:
        return Path(value).expanduser() if value is not None else None
    if name in BOOL_PIPELINE_FIELDS:
        return bool(value)
    if name == "lyric_font_size" or name == "lyric_word_spacing":
        return int(value)
    return value


def _pipeline_options_from_mapping(data: dict[str, Any]) -> PipelineOptions:
    explicit = frozenset(name for name in data if name in RUN_PIPELINE_FIELDS)
    values: dict[str, Any] = {name: None for name in RUN_PIPELINE_FIELDS}
    values["dry_run"] = False
    values["word_level_subtitles"] = False
    for name in RUN_PIPELINE_FIELDS:
        if name not in data:
            continue
        values[name] = _coerce_pipeline_value(name, data[name])
    return PipelineOptions(**values, _explicit=explicit)


def _parse_watch_entry(raw: dict[str, Any], index: int) -> WatchEntry:
    if "directory" not in raw:
        raise ValueError(f"[[watch]] entry {index + 1} is missing required 'directory'")

    directory = Path(str(raw["directory"])).expanduser()
    glob_pattern = str(raw.get("glob", "*"))
    debounce = float(raw.get("debounce", 2.0))
    recursive = bool(raw.get("recursive", True))
    registry_raw = raw.get("registry")
    registry = Path(str(registry_raw)).expanduser() if registry_raw else None

    pipeline_data = {k: v for k, v in raw.items() if k in RUN_PIPELINE_FIELDS}
    pipeline = _pipeline_options_from_mapping(pipeline_data)

    return WatchEntry(
        directory=directory,
        glob=glob_pattern,
        debounce=debounce,
        recursive=recursive,
        registry=registry,
        pipeline=pipeline,
    )


def load_user_config(path: Path | str | None = None) -> UserConfig:
    """
    Load user config from TOML.

    Missing files return an empty config (not an error).
    """
    config_path = resolve_config_path(path)
    if not config_path.exists():
        logger.debug("config file not found: %s", config_path)
        return UserConfig(path=None)

    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read config {config_path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a table: {config_path}")

    defaults = _pipeline_options_from_mapping(raw.get("defaults", {}))

    watches: list[WatchEntry] = []
    watch_tables = raw.get("watch", [])
    if not isinstance(watch_tables, list):
        raise ValueError(f"'watch' must be an array of tables in {config_path}")
    for index, item in enumerate(watch_tables):
        if not isinstance(item, dict):
            raise ValueError(f"[[watch]] entry {index + 1} must be a table")
        entry = _parse_watch_entry(item, index)
        watches.append(
            WatchEntry(
                directory=entry.directory,
                glob=entry.glob,
                debounce=entry.debounce,
                recursive=entry.recursive,
                registry=entry.registry,
                pipeline=defaults.merged_with(entry.pipeline),
            )
        )

    serve_raw = raw.get("serve", {})
    serve = ServeOptions()
    if isinstance(serve_raw, dict) and serve_raw.get("fswatch"):
        serve.fswatch = Path(str(serve_raw["fswatch"])).expanduser()

    logger.debug("loaded config from %s (%d watches)", config_path, len(watches))
    return UserConfig(path=config_path, defaults=defaults, watches=watches, serve=serve)


def apply_defaults_to_args(args: Namespace, defaults: PipelineOptions) -> None:
    """Fill unset CLI pipeline fields from config defaults."""
    for name in RUN_PIPELINE_FIELDS:
        if name in defaults._explicit:
            if name in BOOL_PIPELINE_FIELDS:
                if not getattr(args, name):
                    setattr(args, name, getattr(defaults, name))
            elif getattr(args, name) is None:
                setattr(args, name, getattr(defaults, name))
            continue
        if name in BOOL_PIPELINE_FIELDS:
            continue
        if getattr(args, name) is None and getattr(defaults, name) is not None:
            setattr(args, name, getattr(defaults, name))
