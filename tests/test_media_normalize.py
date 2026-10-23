"""Media container normalisation (fragmented MP4 to progressive faststart)."""

import os
import shutil
import struct
import subprocess

import pytest

from speech_transcribe.mp4 import (
    _scan_needs_remux,
    is_fragmented_mp4,
    is_fragmented_mp4_reader,
    normalize_for_playback,
    normalize_if_fragmented,
)


def _box(box_type: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


_FTYP = _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2mp41")
_MOOV = _box(b"moov", b"\x00" * 16)
_MOOF = _box(b"moof", b"\x00" * 16)
_MDAT = _box(b"mdat", b"\x00" * 64)


def _write(tmp_path, data: bytes) -> str:
    p = os.path.join(tmp_path, "f.mp4")
    with open(p, "wb") as fh:
        fh.write(data)
    return p


# detection (pure, no ffmpeg)

def test_detects_fragmented_mp4(tmp_path):
    p = _write(tmp_path, _FTYP + _MOOV + _MOOF + _MDAT)
    assert is_fragmented_mp4(p) is True


def test_progressive_mp4_is_not_fragmented(tmp_path):
    p = _write(tmp_path, _FTYP + _MOOV + _MDAT)
    assert is_fragmented_mp4(p) is False


def test_non_isobmff_is_not_fragmented(tmp_path):
    # An MP3 (ID3 header) is not ISO-BMFF, never our case, even though it has no moov.
    p = _write(tmp_path, b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb" + b"\x00" * 64)
    assert is_fragmented_mp4(p) is False


def test_truncated_file_is_not_fragmented(tmp_path):
    p = _write(tmp_path, b"\x00\x00")
    assert is_fragmented_mp4(p) is False


def test_normalize_progressive_returns_none(tmp_path):
    p = _write(tmp_path, _FTYP + _MOOV + _MDAT)
    assert normalize_if_fragmented(p) is None


# reader detection (no full download)

def _reader(data: bytes):
    """Range-style read(offset, length) over an in-memory buffer, counting requests."""
    calls = [0]

    def read(offset: int, length: int) -> bytes:
        calls[0] += 1
        return data[offset:offset + length]

    return read, calls


def test_reader_detects_fragmented():
    read, _ = _reader(_FTYP + _MOOV + _MOOF + _MDAT)
    assert is_fragmented_mp4_reader(read) is True


def test_reader_progressive_large_moov_header_only():
    # The decision goes by the box headers alone, never by the content.
    big_moov = _box(b"moov", b"\x00" * 200_000)
    read, calls = _reader(_FTYP + big_moov + _MDAT)
    assert is_fragmented_mp4_reader(read) is False
    assert calls[0] <= 5


# playback check: an index behind the media data needs a remux too

def test_index_behind_media_needs_remux():
    data = _FTYP + _MDAT + _MOOV
    assert _scan_needs_remux(lambda off, n: data[off:off + n]) is True


def test_index_before_media_is_fine(tmp_path):
    assert normalize_for_playback(_write(tmp_path, _FTYP + _MOOV + _MDAT)) is None


def test_fragmented_still_needs_remux():
    data = _FTYP + _MOOV + _MOOF + _MDAT
    assert _scan_needs_remux(lambda off, n: data[off:off + n]) is True


# remux integration (needs ffmpeg and ffprobe)

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_remux_defragments_and_preserves_duration(tmp_path):
    frag = str(tmp_path / "frag.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:a", "aac", "-movflags", "frag_keyframe+empty_moov", "-frag_duration", "400000", frag],
        check=True, capture_output=True,
    )
    assert is_fragmented_mp4(frag) is True
    out = normalize_if_fragmented(frag)
    assert out is not None
    try:
        assert is_fragmented_mp4(out) is False
        assert abs(_duration(out) - _duration(frag)) < 0.2
    finally:
        os.unlink(out)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not available")
def test_moov_at_end_is_moved_to_front(tmp_path):
    src = str(tmp_path / "late.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", src],
        check=True, capture_output=True,
    )
    out = normalize_for_playback(src)
    assert out is not None
    try:
        with open(out, "rb") as fh:
            data = fh.read()
        assert _scan_needs_remux(lambda off, n: data[off:off + n]) is False
    finally:
        os.unlink(out)
