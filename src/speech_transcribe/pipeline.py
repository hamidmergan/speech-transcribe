"""Transcription with chunks: cut when needed, recognise each chunk, shift and merge."""

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import client as stt
from .cache import ChunkCache, NoCache, sweep
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
    empty_chunks: int = 0
    extra: Dict = field(default_factory=dict)


def _reject_if_too_long(duration: float, settings: Settings) -> None:
    limit = int(settings.max_duration_min * 60)
    if duration > limit:
        raise AudioTooLongError(duration, limit)


def _fail_chunk(index: int, error: Exception) -> ModelError:
    """One uniform error message for a failed chunk. Finished chunks stay in the work folder."""
    err = ModelError(f"Chunk {index + 1} transcription failed: {error}")
    err.resumable = True
    return err


def _report(on_progress: ProgressFn, done: int, total: int) -> None:
    """Report the progress, an error there must not stop the run."""
    if on_progress is None:
        return
    try:
        on_progress(done, total)
    except Exception as e:
        logger.warning("Chunk progress could not be reported: %s", e)


def _shift(segments: List[Dict], offset: float, limit: Optional[float] = None) -> List[Dict]:
    """Move a chunk's segments to their place in the recording.

    limit is the end of the chunk. A model that keeps writing past the cut
    would otherwise overlap the first cue of the next chunk.
    """
    shifted = []
    for seg in segments:
        start = seg["start"] + offset
        end = max(seg["end"] + offset, start)
        if limit is not None:
            start, end = min(start, limit), min(end, limit)
        shifted.append({"start": start, "end": end, "text": seg["text"]})
    return shifted


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
    cache_key: Optional[str] = None,
) -> RawResult:
    """Transcribe a recording with timestamps, splitting it when it is longer than one chunk.

    Without a cache_key nothing is written to disk. With one, every finished
    chunk is stored in settings.work_dir under that key, and with resume=True a
    later run takes them from there and only sends the missing chunks.
    """
    normalized = normalize_for_playback(str(file_path))
    media_path = normalized or str(file_path)
    try:
        return _transcribe(media_path, settings, resume, on_progress, client, cache_key)
    finally:
        if normalized:
            try:
                os.unlink(normalized)
            except OSError as e:  # the result is already there, a leftover temp file must not hide it
                logger.warning("Temporary MP4 copy could not be removed: %s", e)


def _transcribe(
    media_path: str,
    settings: Settings,
    resume: bool,
    on_progress: ProgressFn,
    client,
    key: Optional[str],
) -> RawResult:
    duration = get_audio_duration(media_path)
    _reject_if_too_long(duration, settings)
    if client is None:
        client = stt.make_client(settings)

    if duration <= settings.chunk_seconds:
        segs, lang, text = _with_fallback_segment(
            stt.transcribe_verbose(client, settings, media_path), 0.0, duration
        )
        return _finish(_shift(segs, 0.0, duration), duration, lang, text, 1, 0)

    logger.info("Audio %.1fs exceeds the chunk length, splitting into chunks", duration)
    silence_starts = find_silence_starts(media_path)
    split_points = compute_split_points(duration, silence_starts, settings.chunk_seconds)
    ends = list(split_points) + [duration]

    if key:
        # Entries nobody resumed in time go before this run adds its own.
        sweep(settings.work_dir)
        cache = ChunkCache(settings.work_dir, key)
        if not resume:
            cache.clear()
    else:
        cache = NoCache()

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
    empty = 0
    for i in range(total):
        (segs, lang, text), offset = results[i]
        if i == 0:
            first_lang = lang
        if not segs and not text:
            # A chunk that answers with nothing leaves a silent gap in the transcript.
            empty += 1
            logger.warning("Chunk %d/%d came back empty, that part is missing", i + 1, total)
        all_segments.extend(_shift(segs, offset, ends[i]))
        if text:
            texts.append(text)

    cache.clear()
    return _finish(
        all_segments, duration, first_lang, " ".join(texts).strip(), total, resumed, empty
    )


def _finish(segs, duration, lang, text, chunks, resumed, empty=0) -> RawResult:
    if not segs and not text:
        raise ModelError("The server returned no segments for this audio or video file.")
    if not text:
        text = " ".join(s["text"] for s in segs).strip()
    return RawResult(segs, duration, lang, text, chunks, resumed, empty)
