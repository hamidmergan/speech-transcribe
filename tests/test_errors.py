"""The error contract: nothing but a TranscribeError leaves the public entry point."""

import pytest
from conftest import needs_ffmpeg

from speech_transcribe import transcribe
from speech_transcribe.errors import MediaError, TranscribeError


def _args(server, tmp_path):
    return dict(
        base_url=server.base_url,
        model="speech-model",
        work_dir=str(tmp_path / "work"),
    )


@needs_ffmpeg
def test_an_unexpected_error_becomes_a_transcribe_error(server, copy_media, tmp_path, monkeypatch):
    from speech_transcribe import pipeline

    def boom(path):
        raise ValueError("could not convert string to float: 'N/A'")

    monkeypatch.setattr(pipeline, "get_audio_duration", boom)
    with pytest.raises(TranscribeError) as e:
        transcribe(copy_media("short"), **_args(server, tmp_path))
    assert isinstance(e.value.__cause__, ValueError)


@needs_ffmpeg
def test_an_operating_system_error_becomes_a_transcribe_error(
    server, copy_media, tmp_path, monkeypatch
):
    from speech_transcribe import pipeline

    def boom(path):
        raise PermissionError("no access to the temporary folder")

    monkeypatch.setattr(pipeline, "normalize_for_playback", boom)
    with pytest.raises(TranscribeError):
        transcribe(copy_media("short"), **_args(server, tmp_path))


@needs_ffmpeg
def test_a_known_error_keeps_its_own_class(server, copy_media, tmp_path, monkeypatch):
    """The net must not flatten the specific errors into the base class."""
    from speech_transcribe import pipeline

    def boom(path):
        raise MediaError("ffprobe failed")

    monkeypatch.setattr(pipeline, "get_audio_duration", boom)
    with pytest.raises(MediaError):
        transcribe(copy_media("short"), **_args(server, tmp_path))


def test_every_error_class_is_a_transcribe_error():
    from speech_transcribe import errors

    classes = [
        value
        for value in vars(errors).values()
        if isinstance(value, type) and issubclass(value, Exception)
    ]
    assert len(classes) >= 8
    assert all(issubclass(c, TranscribeError) for c in classes)


def test_a_missing_temporary_file_is_a_media_error(monkeypatch):
    from speech_transcribe import mp4

    def boom(**kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(mp4.tempfile, "mkstemp", boom)
    with pytest.raises(MediaError, match="temporary file"):
        mp4._remux_to_temp("talk.mp4")
