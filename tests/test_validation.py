"""Input files are checked by extension, size and leading bytes."""

import pytest
from conftest import needs_ffmpeg

from speech_transcribe import transcribe
from speech_transcribe.errors import LimitError, UnsupportedFileError
from speech_transcribe.validation import EXTENSIONS, check_file

_MP3_HEAD = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" * 8


def _write(tmp_path, name, data=_MP3_HEAD):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_a_missing_file_is_named(tmp_path):
    with pytest.raises(UnsupportedFileError, match="File not found"):
        check_file(tmp_path / "gone.mp3", 500)


def test_an_empty_file_is_refused(tmp_path):
    with pytest.raises(UnsupportedFileError, match="empty"):
        check_file(_write(tmp_path, "empty.mp3", b""), 500)


def test_an_unknown_extension_is_refused(tmp_path):
    with pytest.raises(UnsupportedFileError, match="allowed"):
        check_file(_write(tmp_path, "notes.txt"), 500)


def test_the_common_audio_and_video_types_are_allowed():
    for ext in (".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".webm", ".mpeg"):
        assert ext in EXTENSIONS


def test_a_file_over_the_size_limit_is_refused(tmp_path):
    big = _write(tmp_path, "big.mp3", _MP3_HEAD + b"\x00" * 4096)
    with pytest.raises(LimitError, match="too large"):
        check_file(big, max_file_mb=0.001)
    assert "--max-file-mb" in _message(big)


def _message(path):
    try:
        check_file(path, max_file_mb=0.001)
    except LimitError as e:
        return str(e)
    return ""


def test_a_renamed_file_is_caught(tmp_path):
    fake = _write(tmp_path, "song.mp3", b"%PDF-1.7\n" + b"\x00" * 64)
    with pytest.raises(UnsupportedFileError, match="does not match the extension"):
        check_file(fake, 500)


def test_a_real_mp3_passes(tmp_path):
    assert check_file(_write(tmp_path, "song.mp3"), 500).name == "song.mp3"


def test_the_extension_is_case_insensitive(tmp_path):
    assert check_file(_write(tmp_path, "song.MP3"), 500).name == "song.MP3"


# --- the public function ----------------------------------------------------


@needs_ffmpeg
def test_transcribe_returns_a_transcript(server, copy_media):
    transcript = transcribe(
        copy_media("short"), base_url=server.base_url, model="speech-model", fmt="vtt"
    )
    assert transcript.text == "request 1 second"
    assert transcript.language == "english"
    assert transcript.duration > 10
    assert transcript.chunks == 1
    assert str(transcript).startswith("WEBVTT")


@needs_ffmpeg
def test_transcribe_reports_its_progress(server, copy_media):
    seen = []
    transcribe(
        copy_media("long", "long.wav"),
        base_url=server.base_url,
        model="speech-model",
        chunk_seconds=30,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert sorted(seen) == [(1, 3), (2, 3), (3, 3)]
