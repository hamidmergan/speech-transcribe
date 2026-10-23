"""Validates input files by extension, size and the actual leading bytes."""

from pathlib import Path

import filetype

from .errors import LimitError, UnsupportedFileError

EXTENSIONS = {
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".webm": "video/webm",
    ".mpeg": "video/mpeg",
    ".mpga": "audio/mpeg",
}

# Which detected types match which extension.
_EXT_TO_DETECTED = {
    ".mp3": {"mp3"},
    ".mp4": {"mp4"},
    ".m4a": {"m4a", "mp4"},
    ".wav": {"wav"},
    ".flac": {"flac"},
    ".ogg": {"ogg"},
    ".oga": {"ogg"},
    ".webm": {"webm"},
    ".mpeg": {"mpg"},
    ".mpga": {"mp3"},
}

_HEAD_BYTES = 8192


def validate_content(content: bytes, ext: str) -> None:
    """Check via the leading bytes whether the content matches the extension."""
    expected = _EXT_TO_DETECTED.get(ext.lower())
    if expected is None:
        return
    kind = filetype.guess(content)
    if kind is None or kind.extension not in expected:
        raise UnsupportedFileError(
            "File content does not match the extension. "
            "The file may have been renamed or is corrupted."
        )


def check_file(path, max_file_mb: float) -> Path:
    """Check existence, extension, size and leading bytes. Returns the path."""
    path = Path(path)
    if not path.is_file():
        raise UnsupportedFileError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext not in EXTENSIONS:
        raise UnsupportedFileError(f"Only {', '.join(EXTENSIONS)} allowed")
    size = path.stat().st_size
    if size == 0:
        raise UnsupportedFileError("File is empty")
    if size > max_file_mb * 1024 * 1024:
        raise LimitError(
            f"File too large (max {max_file_mb:g} MB). "
            "Raise max_file_mb (--max-file-mb) to allow larger files."
        )
    with open(path, "rb") as f:
        head = f.read(_HEAD_BYTES)
    validate_content(head, ext)
    return path
