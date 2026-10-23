class TranscribeError(Exception):
    """Base class of all errors in this package."""


class ConfigError(TranscribeError):
    """A setting is missing or a value is invalid."""


class MissingToolError(TranscribeError):
    """A required system program (ffmpeg or ffprobe) is not installed."""


class LimitError(TranscribeError):
    """A limit was exceeded."""


class AudioTooLongError(LimitError):
    """Recording longer than allowed, the limit comes in as an argument."""

    def __init__(self, duration_sec: float, max_sec: int):
        self.duration_sec = duration_sec
        self.max_sec = max_sec
        super().__init__(
            f"Recording too long ({duration_sec / 60:.0f} minutes). "
            f"The maximum is {max_sec // 60} minutes. "
            "Raise max_duration_min (--max-duration-min) to allow longer recordings."
        )


class UnsupportedFileError(TranscribeError):
    """The file type is not supported or the file cannot be read."""


class MediaError(TranscribeError):
    """ffmpeg or ffprobe could not process the file."""


class ModelError(TranscribeError):
    """The speech-to-text server did not answer, or the answer was unusable."""
