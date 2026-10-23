"""Chunked transcription against the fake server: offsets, resume, progress, limits."""

from pathlib import Path

import pytest

from speech_transcribe import config, pipeline
from speech_transcribe.cache import ChunkCache, cache_key
from speech_transcribe.errors import AudioTooLongError, ModelError

CHUNKS = 4
CHUNK_SECONDS = 540.0


@pytest.fixture
def recording(tmp_path) -> str:
    """A placeholder input file. Splitting and probing are stubbed out."""
    path = tmp_path / "lecture.webm"
    path.write_bytes(b"recording bytes")
    return str(path)


@pytest.fixture
def four_chunks(monkeypatch):
    """Deterministic 4-chunk split of a 2000 s recording, without ffmpeg."""
    monkeypatch.setattr(pipeline, "normalize_for_playback", lambda p: None)
    monkeypatch.setattr(pipeline, "get_audio_duration", lambda p: 2000.0)
    monkeypatch.setattr(pipeline, "find_silence_starts", lambda p: [])
    monkeypatch.setattr(pipeline, "compute_split_points", lambda d, s, c: [540.0, 1080.0, 1620.0])

    def fake_split(file_path, split_points, tmpdir):
        chunks = []
        for i in range(len(split_points) + 1):
            chunk = Path(tmpdir) / f"chunk_{i:03d}.mp3"
            chunk.write_bytes(b"chunk audio %d" % i)
            chunks.append((str(chunk), i * CHUNK_SECONDS))
        return chunks

    monkeypatch.setattr(pipeline, "split_audio", fake_split)


def _settings(server, tmp_path, **over):
    values = dict(
        base_url=server.base_url,
        model="speech-model",
        chunk_seconds=CHUNK_SECONDS,
        concurrency=1,  # serial, so request order equals chunk order
        work_dir=str(tmp_path / "work"),
    )
    values.update(over)
    return config.resolve(**values)


def _run(server, tmp_path, recording, **kwargs):
    return pipeline.transcribe_media(recording, _settings(server, tmp_path), **kwargs)


# --- offsets and merging ----------------------------------------------------


def test_every_chunk_is_sent_once(server, tmp_path, recording, four_chunks):
    result = _run(server, tmp_path, recording)
    assert result.chunks == CHUNKS
    assert len(server.requests) == CHUNKS
    assert {r["model"] for r in server.requests} == {"speech-model"}
    assert {r["response_format"] for r in server.requests} == {"verbose_json"}


def test_segments_are_shifted_by_the_chunk_start(server, tmp_path, recording, four_chunks):
    """The fake server always answers 1.0-2.5 s and 5.0-6.0 s of its own chunk."""
    result = _run(server, tmp_path, recording)
    starts = [seg["start"] for seg in result.segments]
    assert starts == [1.0, 5.0, 541.0, 545.0, 1081.0, 1085.0, 1621.0, 1625.0]
    assert [seg["end"] for seg in result.segments][:2] == [2.5, 6.0]


def test_segments_stay_in_recording_order(server, tmp_path, recording, four_chunks):
    result = _run(server, tmp_path, recording)
    assert [seg["text"] for seg in result.segments][::2] == [
        "request 1", "request 2", "request 3", "request 4",
    ]
    assert result.text == "request 1 second request 2 second request 3 second request 4 second"


def test_empty_segments_are_dropped(server, tmp_path, recording, four_chunks):
    """The fake answer carries a whitespace-only segment, it never reaches the output."""
    result = _run(server, tmp_path, recording)
    assert all(seg["text"].strip() for seg in result.segments)
    assert len(result.segments) == CHUNKS * 2


def test_language_and_duration_come_back(server, tmp_path, recording, four_chunks):
    result = _run(server, tmp_path, recording)
    assert result.language == "english"
    assert result.duration == 2000.0


def test_language_is_passed_to_the_server(server, tmp_path, recording, four_chunks):
    settings = _settings(server, tmp_path, language="de")
    pipeline.transcribe_media(recording, settings)
    assert {r["language"] for r in server.requests} == {"de"}


def test_short_recording_is_one_request(server, tmp_path, recording, four_chunks, monkeypatch):
    monkeypatch.setattr(pipeline, "get_audio_duration", lambda p: 60.0)
    result = _run(server, tmp_path, recording)
    assert result.chunks == 1
    assert len(server.requests) == 1
    assert [seg["start"] for seg in result.segments] == [1.0, 5.0]


def test_text_without_segments_becomes_one_segment_per_chunk(
    server, tmp_path, recording, four_chunks
):
    server.no_segments = True
    result = _run(server, tmp_path, recording)
    assert [(s["start"], s["end"]) for s in result.segments] == [
        (0.0, 540.0), (540.0, 1080.0), (1080.0, 1620.0), (1620.0, 2000.0)
    ]


# --- progress ---------------------------------------------------------------


def test_progress_is_reported_once_per_chunk(server, tmp_path, recording, four_chunks):
    seen = []
    _run(server, tmp_path, recording, on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_short_recording_reports_nothing(server, tmp_path, recording, four_chunks, monkeypatch):
    monkeypatch.setattr(pipeline, "get_audio_duration", lambda p: 60.0)
    seen = []
    _run(server, tmp_path, recording, on_progress=lambda d, t: seen.append((d, t)))
    assert seen == []


def test_a_broken_progress_callback_does_not_stop_the_run(server, tmp_path, recording, four_chunks):
    def boom(done, total):
        raise RuntimeError("no screen")

    assert _run(server, tmp_path, recording, on_progress=boom).chunks == CHUNKS


# --- resume -----------------------------------------------------------------


def _cache(server, tmp_path, recording) -> ChunkCache:
    settings = _settings(server, tmp_path)
    return ChunkCache(
        settings.work_dir,
        cache_key(recording, settings.model, settings.language, settings.chunk_seconds),
    )


def test_a_failed_chunk_keeps_the_finished_ones(server, tmp_path, recording, four_chunks):
    server.fail_on = {3}  # the third request, so chunk index 2
    with pytest.raises(ModelError, match="Chunk 3 transcription failed"):
        _run(server, tmp_path, recording)
    cache = _cache(server, tmp_path, recording)
    assert cache.get(0) is not None
    assert cache.get(1) is not None
    assert cache.get(2) is None


def test_the_failure_says_the_run_can_continue(server, tmp_path, recording, four_chunks):
    server.fail_on = {3}
    with pytest.raises(ModelError) as e:
        _run(server, tmp_path, recording)
    assert getattr(e.value, "resumable", False) is True


def test_rerun_with_resume_only_sends_the_missing_chunk(
    server, tmp_path, recording, four_chunks
):
    server.fail_on = {3}
    with pytest.raises(ModelError):
        _run(server, tmp_path, recording)
    sent_before = len(server.requests)
    server.fail_on = set()

    result = _run(server, tmp_path, recording, resume=True)

    assert len(server.requests) == sent_before + 1  # only the chunk that failed
    assert result.resumed == CHUNKS - 1
    assert len(result.segments) == CHUNKS * 2
    assert [seg["start"] for seg in result.segments][:4] == [1.0, 5.0, 541.0, 545.0]


def test_a_successful_run_empties_the_work_folder(server, tmp_path, recording, four_chunks):
    _run(server, tmp_path, recording)
    assert _cache(server, tmp_path, recording).count() == 0


def test_without_resume_everything_is_transcribed_again(server, tmp_path, recording, four_chunks):
    server.fail_on = {3}
    with pytest.raises(ModelError):
        _run(server, tmp_path, recording)
    sent_before = len(server.requests)
    server.fail_on = set()

    result = _run(server, tmp_path, recording)

    assert len(server.requests) == sent_before + CHUNKS
    assert result.resumed == 0


def test_a_changed_setting_does_not_reuse_old_chunks(server, tmp_path, recording, four_chunks):
    _run(server, tmp_path, recording)
    first = _cache(server, tmp_path, recording).dir
    other = ChunkCache(
        _settings(server, tmp_path).work_dir,
        cache_key(recording, "other-model", None, CHUNK_SECONDS),
    )
    assert other.dir != first


# --- retry and limits -------------------------------------------------------


def test_a_temporary_server_error_is_retried(server, tmp_path, recording, four_chunks, monkeypatch):
    from speech_transcribe import client as stt

    monkeypatch.setattr(stt.time, "sleep", lambda s: None)
    server.status_on_fail = 503
    server.fail_on = {1}

    result = _run(server, tmp_path, recording)

    assert result.chunks == CHUNKS
    assert len(server.requests) == CHUNKS + 1  # the refused chunk was sent again


def test_a_permanent_error_is_not_retried(server, tmp_path, recording, four_chunks, monkeypatch):
    from speech_transcribe import client as stt

    waits = []
    monkeypatch.setattr(stt.time, "sleep", lambda s: waits.append(s))
    server.fail_on = {1}  # 400

    with pytest.raises(ModelError):
        _run(server, tmp_path, recording)
    assert waits == []
    assert len(server.requests) == CHUNKS  # every chunk once, no second attempt


def test_a_recording_over_the_limit_is_refused(server, tmp_path, recording, four_chunks, monkeypatch):
    monkeypatch.setattr(pipeline, "get_audio_duration", lambda p: 7000.0)
    settings = _settings(server, tmp_path, max_duration_min=100)
    with pytest.raises(AudioTooLongError) as e:
        pipeline.transcribe_media(recording, settings)
    assert "116.7 minutes" in str(e.value)
    assert "100 minutes" in str(e.value)
    assert server.requests == []


def test_a_recording_just_over_the_limit_does_not_read_as_the_limit(
    server, tmp_path, recording, four_chunks, monkeypatch
):
    """100 min 30 s must not be reported as "too long (100 minutes), the maximum is 100"."""
    monkeypatch.setattr(pipeline, "get_audio_duration", lambda p: 6030.0)
    settings = _settings(server, tmp_path, max_duration_min=100)
    with pytest.raises(AudioTooLongError) as e:
        pipeline.transcribe_media(recording, settings)
    assert "100.5 minutes" in str(e.value)
    assert "The maximum is 100 minutes" in str(e.value)


def test_a_raised_limit_allows_the_long_recording(server, tmp_path, recording, four_chunks, monkeypatch):
    monkeypatch.setattr(pipeline, "get_audio_duration", lambda p: 7000.0)
    settings = _settings(server, tmp_path, max_duration_min=200)
    assert pipeline.transcribe_media(recording, settings).duration == 7000.0


def test_an_answer_without_text_and_segments_is_an_error(
    server, tmp_path, recording, four_chunks, monkeypatch
):
    monkeypatch.setattr(pipeline, "get_audio_duration", lambda p: 60.0)
    monkeypatch.setattr(
        pipeline.stt, "transcribe_verbose", lambda client, settings, path: ([], "", "")
    )
    with pytest.raises(ModelError, match="no segments"):
        _run(server, tmp_path, recording)


def test_a_fragmented_mp4_is_remuxed_before_probing(
    server, tmp_path, recording, four_chunks, monkeypatch
):
    """The temporary copy is probed and removed again."""
    copy = tmp_path / "progressive.mp4"
    copy.write_bytes(b"remuxed")
    probed = []
    monkeypatch.setattr(pipeline, "normalize_for_playback", lambda p: str(copy))
    monkeypatch.setattr(pipeline, "get_audio_duration", lambda p: probed.append(p) or 60.0)

    _run(server, tmp_path, recording)

    assert probed == [str(copy)]
    assert not copy.exists()
