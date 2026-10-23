"""Where a long recording is cut: at the nearest speech pause, always forward."""

import pytest

from speech_transcribe.media import compute_split_points

from conftest import needs_ffmpeg


def test_short_recording_is_not_split():
    assert compute_split_points(300.0, [], chunk_seconds=540) == []


def test_tail_shorter_than_ten_seconds_stays_with_the_last_chunk():
    """A 545 s recording is one chunk, a 5 s rest is not worth its own request."""
    assert compute_split_points(545.0, [], chunk_seconds=540) == []


def test_without_pauses_the_cut_sits_on_the_target():
    assert compute_split_points(2000.0, [], chunk_seconds=540) == [540.0, 1080.0, 1620.0]


def test_cut_moves_to_the_nearest_pause():
    """Pauses at 500 and 560, the one closer to 540 wins."""
    assert compute_split_points(1000.0, [500.0, 560.0], chunk_seconds=540) == [560.0]


def test_pause_outside_the_window_is_ignored():
    """The search window is 60 s, a pause at 400 is too far from 540."""
    assert compute_split_points(1000.0, [400.0], chunk_seconds=540) == [540.0]


def test_next_target_counts_from_the_chosen_pause():
    """After a cut at 500 the next target is 500 + 540, not 1080."""
    points = compute_split_points(1500.0, [500.0, 1040.0], chunk_seconds=540)
    assert points == [500.0, 1040.0]


def test_window_never_reaches_back_past_the_previous_cut():
    """With a short chunk length the window is half a chunk, so no cut repeats."""
    silence = [float(i) for i in range(0, 300, 5)]
    points = compute_split_points(300.0, silence, chunk_seconds=40)
    assert points == sorted(points)
    assert all(after - before > 0 for before, after in zip(points, points[1:]))


def test_every_chunk_stays_within_the_chunk_length_plus_window():
    points = compute_split_points(5000.0, [], chunk_seconds=540)
    bounds = [0.0] + points + [5000.0]
    assert all(end - start <= 540 + 60 for start, end in zip(bounds, bounds[1:]))


@needs_ffmpeg
def test_real_recording_is_cut_at_its_pauses(media):
    """The 75 s test tone has pauses at 8, 18, 28, ... and is cut on them."""
    from speech_transcribe.media import find_silence_starts, get_audio_duration

    path = str(media["long"])
    duration = get_audio_duration(path)
    assert duration == pytest.approx(75.0, abs=0.5)
    starts = find_silence_starts(path)
    assert starts, "silencedetect found no pause in the test tone"
    points = compute_split_points(duration, starts, chunk_seconds=30)
    assert len(points) == 2
    assert points[0] == pytest.approx(28.0, abs=1.0)
    assert points[1] == pytest.approx(58.0, abs=1.0)


@needs_ffmpeg
def test_split_audio_writes_one_mp3_per_piece(media, tmp_path):
    from speech_transcribe.media import split_audio

    chunks = split_audio(str(media["long"]), [28.0, 58.0], str(tmp_path))
    assert [offset for _, offset in chunks] == [0.0, 28.0, 58.0]
    for path, _ in chunks:
        assert path.endswith(".mp3")
        assert open(path, "rb").read(2) != b""


# --- what ffprobe sends back ------------------------------------------------


def _fake_probe(monkeypatch, stdout):
    """Answer every ffprobe call with the given JSON."""
    from speech_transcribe import media

    class _Done:
        returncode = 0
        stderr = ""

        def __init__(self, out):
            self.stdout = out

    monkeypatch.setattr(media.subprocess, "run", lambda *a, **k: _Done(stdout))


def test_an_unreadable_stream_duration_is_a_media_error(monkeypatch):
    """ffprobe writes "N/A" for a stream it cannot measure."""
    from speech_transcribe.errors import MediaError
    from speech_transcribe.media import get_audio_duration

    _fake_probe(monkeypatch, '{"streams": [{"duration": "N/A"}], "format": {}}')
    with pytest.raises(MediaError, match="unreadable duration"):
        get_audio_duration("talk.mp4")


def test_an_unreadable_container_duration_is_a_media_error(monkeypatch):
    from speech_transcribe.errors import MediaError
    from speech_transcribe.media import get_audio_duration

    _fake_probe(monkeypatch, '{"streams": [], "format": {"duration": "N/A"}}')
    with pytest.raises(MediaError, match="unreadable duration"):
        get_audio_duration("talk.mp4")


def test_a_readable_duration_still_comes_through(monkeypatch):
    from speech_transcribe.media import get_audio_duration

    _fake_probe(monkeypatch, '{"streams": [{"duration": "12.5"}], "format": {}}')
    assert get_audio_duration("talk.mp4") == 12.5


def test_a_missing_ffprobe_is_a_media_error(monkeypatch):
    from speech_transcribe import media
    from speech_transcribe.errors import MediaError

    def boom(*args, **kwargs):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(media.subprocess, "run", boom)
    with pytest.raises(MediaError, match="could not be started"):
        media.get_audio_duration("talk.mp4")
