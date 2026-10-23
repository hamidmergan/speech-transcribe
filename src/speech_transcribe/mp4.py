"""Rewrites fragmented MP4 losslessly into continuously playable files."""

import logging
import os
import subprocess
import tempfile
from typing import Optional

from .errors import MediaError

logger = logging.getLogger(__name__)

# The reader only reads box headers, never the media content.
_MAX_BOXES = 100_000  # safety bound against a malformed or looping file


def _scan_is_fragmented(read) -> bool:
    """Walk the boxes, True at the first moof, False at the first mdat."""
    head = read(0, 8)
    if len(head) < 8 or head[4:8] != b"ftyp":
        return False  # not ISO-BMFF, not our case
    pos = 0
    for _ in range(_MAX_BOXES):
        hdr = read(pos, 16)
        if len(hdr) < 8:
            return False
        box_size = int.from_bytes(hdr[:4], "big")
        box_type = hdr[4:8]
        if box_type == b"moof":
            return True
        if box_type == b"mdat":
            return False
        if box_size == 1:  # 64-bit extended size follows the 8-byte header
            if len(hdr) < 16:
                return False
            box_size = int.from_bytes(hdr[8:16], "big")
        if box_size < 8:  # 0 (extends to EOF) or malformed, stop
            return False
        pos += box_size
    return False


def _scan_needs_remux(read) -> bool:
    """True for a fragmented MP4 and for one whose index sits behind the media data."""
    head = read(0, 8)
    if len(head) < 8 or head[4:8] != b"ftyp":
        return False
    pos = 0
    seen_moov = False
    for _ in range(_MAX_BOXES):
        hdr = read(pos, 16)
        if len(hdr) < 8:
            return False
        box_size = int.from_bytes(hdr[:4], "big")
        box_type = hdr[4:8]
        if box_type == b"moof":
            return True
        if box_type == b"moov":
            seen_moov = True
        if box_type == b"mdat":
            return not seen_moov
        if box_size == 1:
            if len(hdr) < 16:
                return False
            box_size = int.from_bytes(hdr[8:16], "big")
        if box_size < 8:
            return False
        pos += box_size
    return False


def _file_reader(f):
    def _read(off: int, length: int) -> bytes:
        f.seek(off)
        return f.read(length)

    return _read


def is_fragmented_mp4(path: str) -> bool:
    """True when the file is a fragmented MP4."""
    try:
        with open(path, "rb") as f:
            return _scan_is_fragmented(_file_reader(f))
    except OSError:
        return False


def is_fragmented_mp4_reader(read) -> bool:
    """Detect fragmentation through a read(offset, length) function, so it works remotely too."""
    try:
        return _scan_is_fragmented(read)
    except Exception:
        return False


def remux_to_progressive(src: str, dst: str) -> None:
    """Losslessly remux src into a progressive faststart MP4 at dst (no re-encode)."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)],
        capture_output=True,
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace")[:300]
        raise MediaError(f"ffmpeg remux failed: {err}")


def _remux_to_temp(path: str) -> str:
    fd, out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        remux_to_progressive(path, out)
    except Exception:
        os.unlink(out)
        raise
    return out


def normalize_if_fragmented(path: str) -> Optional[str]:
    """Rewrite a fragmented MP4 and return the path of the copy, None otherwise."""
    if not is_fragmented_mp4(path):
        return None
    out = _remux_to_temp(path)
    logger.info("Remuxed fragmented MP4 to progressive faststart (%s)", path)
    return out


def normalize_for_playback(path: str) -> Optional[str]:
    """Rewrite an MP4 that cannot start playing at once, None when it already can."""
    try:
        with open(path, "rb") as f:
            needed = _scan_needs_remux(_file_reader(f))
    except OSError:
        return None
    if not needed:
        return None
    out = _remux_to_temp(path)
    logger.info("Remuxed MP4 to progressive faststart (%s)", path)
    return out
