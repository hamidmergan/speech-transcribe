class TranscribeError(Exception):
    """Base class of all errors in this package."""


class ConfigError(TranscribeError):
    """A setting is missing or a value is invalid."""


class MissingToolError(TranscribeError):
    """A required system program (ffmpeg or ffprobe) is not installed."""
