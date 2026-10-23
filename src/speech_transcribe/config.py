"""Default values and resolution of settings.

Order: argument, then environment variable, then default.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import ConfigError

# Limits
MAX_FILE_MB = 500  # size of one input file
MAX_DURATION_MIN = 100  # length of one recording

# Processing
CHUNK_SECONDS = 540  # 9 min, safely under the typical 600 s upload limit
CONCURRENCY = 2  # chunks sent to the server at once
TIMEOUT_SECONDS = 300  # read timeout per request

# Placeholder, because the OpenAI library requires a value.
# Local servers ignore it.
PLACEHOLDER_API_KEY = "none"

ENV_PREFIX = "SPEECH_TRANSCRIBE_"


def default_work_dir() -> Path:
    """Folder that keeps finished chunks so an interrupted run can resume."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "speech-transcribe"


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str
    language: Optional[str]
    chunk_seconds: float
    max_file_mb: float
    max_duration_min: float
    concurrency: int
    timeout: float
    work_dir: Path


def _env(name: str) -> Optional[str]:
    value = os.environ.get(ENV_PREFIX + name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _pick(arg, env_name: str, default, cast):
    if arg is not None:
        return arg
    raw = _env(env_name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise ConfigError(
            f"Environment variable {ENV_PREFIX}{env_name} has an invalid value: {raw!r}"
        )


def resolve(
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    language: Optional[str] = None,
    chunk_seconds: Optional[float] = None,
    max_file_mb: Optional[float] = None,
    max_duration_min: Optional[float] = None,
    concurrency: Optional[int] = None,
    timeout: Optional[float] = None,
    work_dir=None,
) -> Settings:
    base_url = _pick(base_url, "BASE_URL", None, str)
    model = _pick(model, "MODEL", None, str)
    missing = []
    if not base_url:
        missing.append("base_url (--base-url or SPEECH_TRANSCRIBE_BASE_URL)")
    if not model:
        missing.append("model (--model or SPEECH_TRANSCRIBE_MODEL)")
    if missing:
        raise ConfigError(
            "Missing: " + " and ".join(missing) + ". "
            "Example: base_url http://localhost:11434/v1 and a speech-to-text model."
        )

    key = _pick(api_key, "API_KEY", None, str)
    if not key:
        key = os.environ.get("OPENAI_API_KEY") or PLACEHOLDER_API_KEY

    settings = Settings(
        base_url=base_url,
        model=model,
        api_key=key,
        language=_pick(language, "LANGUAGE", None, str) or None,
        chunk_seconds=_pick(chunk_seconds, "CHUNK_SECONDS", CHUNK_SECONDS, float),
        max_file_mb=_pick(max_file_mb, "MAX_FILE_MB", MAX_FILE_MB, float),
        max_duration_min=_pick(max_duration_min, "MAX_DURATION_MIN", MAX_DURATION_MIN, float),
        concurrency=_pick(concurrency, "CONCURRENCY", CONCURRENCY, int),
        timeout=_pick(timeout, "TIMEOUT", TIMEOUT_SECONDS, float),
        work_dir=Path(_pick(work_dir, "WORK_DIR", None, str) or default_work_dir()),
    )
    _validate(settings)
    return settings


def _validate(s: Settings) -> None:
    positive = {
        "chunk_seconds": s.chunk_seconds,
        "max_file_mb": s.max_file_mb,
        "max_duration_min": s.max_duration_min,
        "concurrency": s.concurrency,
        "timeout": s.timeout,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ConfigError(f"{name} must be greater than 0, but is {value}.")
    if s.chunk_seconds < 30:
        raise ConfigError(f"chunk_seconds must be at least 30, but is {s.chunk_seconds}.")
