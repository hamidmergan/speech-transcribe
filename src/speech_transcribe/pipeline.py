"""Transcription with chunks: cut when needed, recognise each chunk, shift and merge."""

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import client as stt
from .cache import ChunkCache, cache_key
from .config import Settings
from .errors import AudioTooLongError, ModelError, TranscribeError
from .media import compute_split_points, find_silence_starts, get_audio_duration, split_audio
from .mp4 import normalize_for_playback

logger = logging.getLogger(__name__)

# Reports the progress after every finished chunk: (done, total).
ProgressFn = Optional[Callable[[int, int], None]]


@dataclass
class RawResult:
    segments: List[Dict]
    duration: float
    language: str
    text: str
    chunks: int = 1
    resumed: int = 0
    extra: Dict = field(default_factory=dict)


def _reject_if_too_long(duration: float, settings: Settings) -> None:
    limit = int(settings.max_duration_min * 60)
    if duration > limit:
        raise AudioTooLongError(duration, limit)


def _fail_chunk(index: int, error: Exception) -> ModelError:
    """One uniform error message for a failed chunk."""
    return ModelError(f"Chunk {index + 1} transcription failed: {error}")


def _report(on_progress: ProgressFn, done: int, total: int) -> None:
    """Report the progress, an error there must not stop the run."""
    if on_progress is None:
        return
    try:
        on_progress(done, total)
    except Exception as e:
        logger.warning("Chunk progress could not be reported: %s", e)


def _shift(segments: List[Dict], offset: float) -> List[Dict]:
    return [
        {"start": seg["start"] + offset, "end": seg["end"] + offset, "text": seg["text"]}
        for seg in segments
    ]


def _with_fallback_segment(result, start: float, end: float):
    """A server that sends text without segments still yields one segment for the chunk."""
    segs, lang, text = result
    if not segs and text:
        segs = [{"start": 0.0, "end": max(end - start, 0.0), "text": text}]
    return segs, lang, text


def transcribe_media(
    file_path: str,
    settings: Settings,
    *,
    resume: bool = False,
    on_progress: ProgressFn = None,
    client=None,
) -> RawResult:
    """Transcribe a recording with timestamps, splitting it when it is longer than one chunk.

    Finished chunks are stored in settings.work_dir. With resume=True a later run
    takes them from there and only sends the missing chunks.
    """
    normalized = normalize_for_playback(str(file_path))
    media_path = normalized or str(file_path)
    try:
        return _transcribe(str(file_path), media_path, settings, resume, on_progress, client)
    finally:
        if normalized:
            os.unlink(normalized)


def _transcribe(
    original: str,
    media_path: str,
    settings: Settings,
    resume: bool,
    on_progress: ProgressFn,
    client,
) -> RawResult:
    duration = get_audio_duration(media_path)
    _reject_if_too_long(duration, settings)
    if client is None:
        client = stt.make_client(settings)

    if duration <= settings.chunk_seconds:
        segs, lang, text = _with_fallback_segment(
            stt.transcribe_verbose(client, settings, media_path), 0.0, duration
        )
        return _finish(segs, duration, lang, text, 1, 0)

    logger.info("Audio %.1fs exceeds the chunk length, splitting into chunks", duration)
    silence_starts = find_silence_starts(media_path)
    split_points = compute_split_points(duration, silence_starts, settings.chunk_seconds)
    ends = list(split_points) + [duration]

    cache = ChunkCache(
        settings.work_dir,
        cache_key(original, settings.model, settings.language, settings.chunk_seconds),
    )
    if not resume:
        cache.clear()

    def _transcribe_or_cached(path: str, idx: int, start: float, total: int):
        hit = cache.get(idx)
        if hit is not None:
            logger.info("Chunk %d/%d taken from the work folder", idx + 1, total)
            return hit, True
        result = _with_fallback_segment(stt.transcribe_verbose(client, settings, path), start, ends[idx])
        cache.put(idx, *result)
        return result, False

    with tempfile.TemporaryDirectory() as tmpdir:
        chunks = split_audio(media_path, split_points, tmpdir)
        total = len(chunks)
        results: Dict[int, Tuple[Tuple[List[Dict], str, str], float]] = {}
        resumed = 0
        failure: Optional[TranscribeError] = None
        with ThreadPoolExecutor(max_workers=settings.concurrency) as executor:
            futures = {
                executor.submit(_transcribe_or_cached, path, i, offset, total): (i, offset)
                for i, (path, offset) in enumerate(chunks)
            }
            for future in as_completed(futures):
                i, offset = futures[future]
                try:
                    result, from_cache = future.result()
                except Exception as e:
                    # The other chunks keep running and land in the work folder.
                    if failure is None:
                        failure = _fail_chunk(i, e)
                        failure.__cause__ = e
                    continue
                results[i] = (result, offset)
                resumed += int(from_cache)
                logger.info("Chunk %d/%d transcribed", i + 1, total)
                _report(on_progress, len(results), total)
        if failure is not None:
            raise failure

    all_segments: List[Dict] = []
    texts: List[str] = []
    first_lang = ""
    for i in range(total):
        (segs, lang, text), offset = results[i]
        if i == 0:
            first_lang = lang
        all_segments.extend(_shift(segs, offset))
        if text:
            texts.append(text)

    cache.clear()
    return _finish(all_segments, duration, first_lang, " ".join(texts).strip(), total, resumed)


def _finish(segs, duration, lang, text, chunks, resumed) -> RawResult:
    if not segs and not text:
        raise ModelError("The server returned no segments for this audio or video file.")
    if not text:
        text = " ".join(s["text"] for s in segs).strip()
    return RawResult(segs, duration, lang, text, chunks, resumed)
