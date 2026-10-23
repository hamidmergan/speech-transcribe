"""Checks that the system programs ffmpeg and ffprobe are available."""

import shutil

from .errors import MissingToolError

REQUIRED_TOOLS = ("ffmpeg", "ffprobe")


def check_tools() -> None:
    """Raise MissingToolError if ffmpeg or ffprobe is not on the PATH."""
    missing = [name for name in REQUIRED_TOOLS if shutil.which(name) is None]
    if missing:
        raise MissingToolError(
            "Missing system program: " + ", ".join(missing) + ". "
            "Install ffmpeg (it includes ffprobe), for example with "
            "'apt install ffmpeg' or 'brew install ffmpeg'."
        )
