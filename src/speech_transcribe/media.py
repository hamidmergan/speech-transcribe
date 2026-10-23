"""Splits long recordings at speech pauses for the upload."""

import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .config import CHUNK_SECONDS
from .errors import MediaError

CHUNK_REENCODE_BITRATE_KBPS = 64  # 64 kbps mono MP3, about 4 MB per 9 min chunk, stays under upload limits
CHUNK_REENCODE_SAMPLE_RATE_HZ = 16000  # Whisper resamples to 16 kHz internally, match it to skip server-side conversion
_SILENCE_NOISE_DB = -40  # dB threshold for silence detection
_SILENCE_MIN_DURATION = 0.3  # minimum silence duration (s)
_SPLIT_SEARCH_WINDOW = 60  # search window around each target split point (s)


def get_audio_duration(file_path: str) -> float:
    """Return audio or video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(file_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed: {result.stderr[:200]}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise MediaError("ffprobe returned no readable output")
    for stream in data.get("streams", []):
        duration = stream.get("duration")
        if duration:
            return float(duration)
    fmt_duration = data.get("format", {}).get("duration")
    if fmt_duration:
        return float(fmt_duration)
    raise MediaError("Could not determine audio duration via ffprobe")


def find_silence_starts(file_path: str) -> List[float]:
    """Find the speech pauses via ffmpeg, without decoding the video track."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(file_path),
            "-vn",
            "-af", f"silencedetect=noise={_SILENCE_NOISE_DB}dB:d={_SILENCE_MIN_DURATION}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    output = result.stderr + result.stdout
    return [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", output)]


def compute_split_points(
    duration: float, silence_starts: List[float], chunk_seconds: float = CHUNK_SECONDS
) -> List[float]:
    """Return split timestamps aligned to nearby silence points."""
    # The window never reaches back past the previous split, so every step moves forward.
    window = min(_SPLIT_SEARCH_WINDOW, chunk_seconds / 2)
    split_points: List[float] = []
    target = chunk_seconds
    while target < duration - 10:
        window_min = target - window
        window_max = target + window
        candidates = [s for s in silence_starts if window_min <= s <= window_max]
        best = min(candidates, key=lambda s: abs(s - target)) if candidates else target
        split_points.append(best)
        target = best + chunk_seconds
    return split_points


def split_audio(file_path: str, split_points: List[float], tmpdir: str) -> List[Tuple[str, float]]:
    """Cut the recording and turn every piece into a small MP3. Returns (path, start) pairs."""
    boundaries: List[Optional[float]] = [0.0] + list(split_points) + [None]  # None = end of file
    chunks: List[Tuple[str, float]] = []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        chunk_path = str(Path(tmpdir) / f"chunk_{i:03d}.mp3")
        cmd = ["ffmpeg", "-ss", str(start), "-i", str(file_path)]
        if end is not None:
            cmd += ["-t", str(end - start)]  # duration, not absolute end (input seek is relative)
        cmd += [
            "-vn",  # drop video stream
            "-c:a", "libmp3lame",
            "-b:a", f"{CHUNK_REENCODE_BITRATE_KBPS}k",
            "-ac", "1",  # mono
            "-ar", str(CHUNK_REENCODE_SAMPLE_RATE_HZ),
            "-y", chunk_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise MediaError(f"ffmpeg split failed for chunk {i}: {result.stderr[:200]!r}")
        chunks.append((chunk_path, start))
    return chunks
