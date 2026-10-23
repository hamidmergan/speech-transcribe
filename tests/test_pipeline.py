"""Chunked transcription against the fake server: offsets, resume, progress, limits."""

import os
import time
from pathlib import Path

import pytest

from speech_transcribe import cache as cache_module
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


def _key(settings, recording) -> str:
    return cache_key(
        recording, settings.base_url, settings.model, settings.language, settings.chunk_seconds
    )


def _run(server, tmp_path, recording, **kwargs):
    """A run with a cache key, the way the public transcribe() calls the pipeline."""
    settings = _settings(server, tmp_path)
    kwargs.setdefault("cache_key", _key(settings, recording))
    return pipeline.transcribe_media(recording, settings, **kwargs)


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
    return ChunkCache(settings.work_dir, _key(settings, recording))


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
    # the cached texts have to land in recording order, not in the order they came back
    assert result.text == (
        "request 1 second request 2 second request 5 second request 4 second"
    )
    # a resumed run that succeeds empties the work folder as well
    assert _cache(server, tmp_path, recording).count() == 0


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
        cache_key(recording, server.base_url, "other-model", None, CHUNK_SECONDS),
    )
    assert other.dir != first


def test_another_server_does_not_reuse_old_chunks(server, tmp_path, recording):
    """Two servers may both call their model whisper-1."""
    mine = cache_key(recording, "http://a/v1", "whisper-1", None, CHUNK_SECONDS)
    theirs = cache_key(recording, "http://b/v1", "whisper-1", None, CHUNK_SECONDS)
    assert mine != theirs


# --- caching is opt in and does not last for ever ---------------------------


def test_no_cache_key_means_no_caching(server, tmp_path, recording, four_chunks):
    """Without a key the run writes nothing to disk, not even on a failure."""
    server.fail_on = {3}
    settings = _settings(server, tmp_path)
    with pytest.raises(ModelError):
        pipeline.transcribe_media(recording, settings)
    assert not list(Path(settings.work_dir).glob("**/chunk_*.json"))


def test_a_run_with_a_key_keeps_the_finished_chunks(server, tmp_path, recording, four_chunks):
    server.fail_on = {3}
    with pytest.raises(ModelError):
        _run(server, tmp_path, recording)
    settings = _settings(server, tmp_path)
    assert list(Path(settings.work_dir).glob("**/chunk_*.json"))


def test_an_old_chunk_is_not_used_again(server, tmp_path, recording, four_chunks):
    """A cached chunk older than the maximum age counts as absent."""
    server.fail_on = {3}
    with pytest.raises(ModelError):
        _run(server, tmp_path, recording)
    cache = _cache(server, tmp_path, recording)
    assert cache.get(0) is not None
    stale = time.time() - cache_module.MAX_AGE_SECONDS - 60
    for path in cache.dir.glob("chunk_*.json"):
        os.utime(path, (stale, stale))
    assert cache.get(0) is None


def test_old_chunk_folders_are_swept_away(server, tmp_path, recording, four_chunks):
    server.fail_on = {3}
    with pytest.raises(ModelError):
        _run(server, tmp_path, recording)
    settings = _settings(server, tmp_path)
    stale = time.time() - cache_module.MAX_AGE_SECONDS - 60
    for path in Path(settings.work_dir).glob("**/chunk_*.json"):
        os.utime(path, (stale, stale))

    assert cache_module.sweep(settings.work_dir) == 1
    assert not list(Path(settings.work_dir).glob("**/chunk_*.json"))


def test_a_fresh_chunk_folder_survives_the_sweep(server, tmp_path, recording, four_chunks):
    server.fail_on = {3}
    with pytest.raises(ModelError):
        _run(server, tmp_path, recording)
    settings = _settings(server, tmp_path)
    assert cache_module.sweep(settings.work_dir) == 0
    assert list(Path(settings.work_dir).glob("**/chunk_*.json"))


def test_the_next_run_sweeps_what_nobody_resumed(server, tmp_path, recording, four_chunks):
    """An abandoned run must not leave transcript text behind for ever."""
    settings = _settings(server, tmp_path)
    abandoned = ChunkCache(settings.work_dir, "0" * 32)
    abandoned.put(0, [{"start": 0.0, "end": 1.0, "text": "old talk"}], "english", "old talk")
    stale = time.time() - cache_module.MAX_AGE_SECONDS - 60
    for path in abandoned.dir.glob("chunk_*.json"):
        os.utime(path, (stale, stale))

    _run(server, tmp_path, recording)

    assert not abandoned.dir.exists()


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


# --- a chunk that answers with nothing --------------------------------------


def test_an_empty_chunk_is_counted(server, tmp_path, recording, four_chunks):
    """A chunk with an empty 200 leaves a gap, the run must not stay silent."""
    server.empty_on = {2}
    result = _run(server, tmp_path, recording)
    assert result.empty_chunks == 1
    assert result.chunks == CHUNKS
    assert len(result.segments) == (CHUNKS - 1) * 2


def test_a_full_run_counts_no_empty_chunk(server, tmp_path, recording, four_chunks):
    assert _run(server, tmp_path, recording).empty_chunks == 0


# --- a chunk never reaches past its own cut ---------------------------------


def test_a_segment_past_the_cut_is_clamped(server, tmp_path, recording, four_chunks, monkeypatch):
    """A model that keeps writing past the cut must not overlap the next chunk."""
    monkeypatch.setattr(
        pipeline.stt,
        "transcribe_verbose",
        lambda client, settings, path: ([{"start": 500.0, "end": 620.0, "text": "over"}], "en", ""),
    )
    result = _run(server, tmp_path, recording)
    ends = [seg["end"] for seg in result.segments]
    starts = [seg["start"] for seg in result.segments]
    assert ends == [540.0, 1080.0, 1620.0, 2000.0]
    assert all(end <= later for end, later in zip(ends, starts[1:]))


def test_a_single_chunk_stays_inside_the_recording(
    server, tmp_path, recording, four_chunks, monkeypatch
):
    monkeypatch.setattr(pipeline, "get_audio_duration", lambda p: 60.0)
    monkeypatch.setattr(
        pipeline.stt,
        "transcribe_verbose",
        lambda client, settings, path: ([{"start": 10.0, "end": 400.0, "text": "over"}], "en", ""),
    )
    assert [s["end"] for s in _run(server, tmp_path, recording).segments] == [60.0]
