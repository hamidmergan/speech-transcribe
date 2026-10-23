from .core import FORMATS, Transcript, transcribe
from .errors import (
    AudioTooLongError,
    ConfigError,
    LimitError,
    MediaError,
    MissingToolError,
    ModelError,
    TranscribeError,
    UnsupportedFileError,
)
from .mp4 import (
    is_fragmented_mp4,
    is_fragmented_mp4_reader,
    normalize_for_playback,
    normalize_if_fragmented,
)
from .subtitles import segments_to_srt, segments_to_vtt, split_long_segments, vtt_to_segments
from .tools import check_tools

__version__ = "0.1.0"

__all__ = [
    "transcribe",
    "Transcript",
    "FORMATS",
    "check_tools",
    "segments_to_vtt",
    "segments_to_srt",
    "vtt_to_segments",
    "split_long_segments",
    "is_fragmented_mp4",
    "is_fragmented_mp4_reader",
    "normalize_if_fragmented",
    "normalize_for_playback",
    "TranscribeError",
    "ConfigError",
    "LimitError",
    "AudioTooLongError",
    "MediaError",
    "MissingToolError",
    "ModelError",
    "UnsupportedFileError",
    "__version__",
]
