"""Public API: transcribe a recording and render the result."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .cache import cache_key
from .errors import ConfigError, TranscribeError
from .pipeline import ProgressFn, transcribe_media
from .subtitles import segments_to_srt, segments_to_vtt, split_long_segments
from .tools import check_tools
from .validation import check_file

FORMATS = ("txt", "vtt", "srt", "json")
EXTENSIONS = {"txt": ".txt", "vtt": ".vtt", "srt": ".srt", "json": ".json"}
_ALIASES = {"text": "txt"}


def normalize_format(fmt: Optional[str]) -> str:
    """Return the canonical format name or raise ConfigError."""
    name = (fmt or "txt").strip().lower()
    name = _ALIASES.get(name, name)
    if name not in FORMATS:
        raise ConfigError(f"Invalid format: {fmt}. Use one of: {', '.join(FORMATS)}.")
    return name


@dataclass
class Transcript:
    """The result of one recording. Times are in seconds."""

    text: str
    segments: List[Dict] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0
    format: str = "txt"
    chunks: int = 1
    resumed_chunks: int = 0
    empty_chunks: int = 0  # pieces the server answered with nothing, a gap in the text

    def _cues(self, short_cues: bool) -> List[Dict]:
        return split_long_segments(self.segments) if short_cues else self.segments

    def to_text(self) -> str:
        return self.text

    def to_vtt(self, short_cues: bool = False) -> str:
        """WebVTT. short_cues splits long segments into cues of at most 2 lines of 42 characters."""
        return segments_to_vtt(self._cues(short_cues))

    def to_srt(self, short_cues: bool = False) -> str:
        """SRT. short_cues splits long segments into cues of at most 2 lines of 42 characters."""
        return segments_to_srt(self._cues(short_cues))

    def to_dict(self, short_cues: bool = False) -> Dict:
        """text, language, duration and the segments. short_cues splits long segments."""
        return {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in self._cues(short_cues)
            ],
        }

    def to_json(self, short_cues: bool = False) -> str:
        return json.dumps(self.to_dict(short_cues), ensure_ascii=False, indent=2)

    def render(self, fmt: Optional[str] = None, short_cues: bool = False) -> str:
        """The result in the given format, or in the format chosen at transcription."""
        name = normalize_format(fmt or self.format)
        if name == "vtt":
            return self.to_vtt(short_cues)
        if name == "srt":
            return self.to_srt(short_cues)
        if name == "json":
            return self.to_json(short_cues)
        return self.to_text()

    def __str__(self) -> str:
        return self.render()


def transcribe(
    path,
    language: Optional[str] = None,
    fmt: str = "txt",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    resume: bool = False,
    on_progress: ProgressFn = None,
    chunk_seconds: Optional[float] = None,
    max_file_mb: Optional[float] = None,
    max_duration_min: Optional[float] = None,
    concurrency: Optional[int] = None,
    timeout: Optional[float] = None,
    work_dir=None,
) -> Transcript:
    """Transcribe an audio or video file into a Transcript with timestamps.

    language is an optional ISO 639-1 code (for example "en"). Without it the
    server detects the language. fmt picks what str(transcript) and
    transcript.render() return: txt, vtt, srt or json.

    Every failure leaves this function as a TranscribeError, so a caller never
    has to guard against the error types of ffmpeg, the file system or the
    HTTP library.
    """
    try:
        return _transcribe(
            path, language, fmt, base_url, model, api_key,
            resume=resume, on_progress=on_progress, chunk_seconds=chunk_seconds,
            max_file_mb=max_file_mb, max_duration_min=max_duration_min,
            concurrency=concurrency, timeout=timeout, work_dir=work_dir,
        )
    except TranscribeError:
        raise
    except Exception as exc:  # the contract above holds for every error
        raise TranscribeError(f"Transcription of {path} failed: {exc}") from exc


def _transcribe(
    path,
    language,
    fmt,
    base_url,
    model,
    api_key,
    *,
    resume,
    on_progress,
    chunk_seconds,
    max_file_mb,
    max_duration_min,
    concurrency,
    timeout,
    work_dir,
) -> Transcript:
    fmt = normalize_format(fmt)
    settings = config.resolve(
        base_url=base_url,
        model=model,
        api_key=api_key,
        language=language,
        chunk_seconds=chunk_seconds,
        max_file_mb=max_file_mb,
        max_duration_min=max_duration_min,
        concurrency=concurrency,
        timeout=timeout,
        work_dir=work_dir,
    )
    check_tools()
    file_path = check_file(path, settings.max_file_mb)
    raw = transcribe_media(
        str(file_path),
        settings,
        resume=resume,
        on_progress=on_progress,
        cache_key=cache_key(
            file_path, settings.base_url, settings.model, settings.language, settings.chunk_seconds
        ),
    )
    return Transcript(
        text=raw.text,
        segments=raw.segments,
        language=raw.language,
        duration=raw.duration,
        format=fmt,
        chunks=raw.chunks,
        resumed_chunks=raw.resumed,
        empty_chunks=raw.empty_chunks,
    )


def output_name(path, fmt: str) -> str:
    """Name of the result file for an input, for example talk.mp4 -> talk.vtt."""
    return Path(path).stem + EXTENSIONS[normalize_format(fmt)]
