"""Remembers the result per chunk on disk, so a second attempt repeats nothing."""

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ChunkResult = Tuple[List[Dict], str, str]  # segments, language, text

# Chunk files hold transcript text, so they do not stay on disk for ever.
MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days


def cache_key(path, base_url: str, model: str, language: Optional[str], chunk_seconds: float) -> str:
    """Key from the file content and everything that changes the result."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    # Two servers may both call their model "whisper-1", so the address belongs in the key.
    digest.update(f"\0{base_url}\0{model}\0{language or ''}\0{chunk_seconds:g}".encode("utf-8"))
    return digest.hexdigest()[:32]


def _is_stale(path: Path, max_age: float, now: float) -> bool:
    try:
        return now - path.stat().st_mtime > max_age
    except OSError:
        return False


def sweep(work_dir, max_age: float = MAX_AGE_SECONDS) -> int:
    """Delete chunk folders that nobody resumed in time. Returns how many went.

    A run that fails and is never retried would otherwise leave its transcript
    text in the cache folder for ever.
    """
    removed = 0
    now = time.time()
    try:
        entries = list(Path(work_dir).iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            files = list(entry.glob("chunk_*.json"))
            if not files or any(not _is_stale(f, max_age, now) for f in files):
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        except OSError:
            continue
    return removed


class ChunkCache:
    """One folder per recording, one JSON file per finished chunk. Best effort."""

    def __init__(self, work_dir, key: str, max_age: float = MAX_AGE_SECONDS):
        self.dir = Path(work_dir) / key
        self.max_age = max_age

    def _file(self, index: int) -> Path:
        return self.dir / f"chunk_{index:03d}.json"

    def get(self, index: int) -> Optional[ChunkResult]:
        """Return the cached (segments, language, text) for a chunk, or None.

        An entry older than max_age counts as absent and is removed.
        """
        path = self._file(index)
        try:
            if _is_stale(path, self.max_age, time.time()):
                path.unlink(missing_ok=True)
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["segments"], data.get("language", ""), data.get("text", "")
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, index: int, segments: List[Dict], language: str, text: str) -> None:
        """Store a chunk result. Errors are swallowed, the cache only saves time."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            target = self._file(index)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"segments": segments, "language": language, "text": text}),
                encoding="utf-8",
            )
            tmp.replace(target)
        except OSError:
            pass

    def count(self) -> int:
        """Number of finished chunks on disk."""
        try:
            return len(list(self.dir.glob("chunk_*.json")))
        except OSError:
            return 0

    def clear(self) -> None:
        """Drop all chunk entries. Called once a recording is fully transcribed."""
        shutil.rmtree(self.dir, ignore_errors=True)


class NoCache:
    """Used when no cache key was given: nothing is written, nothing is read."""

    dir = None

    def get(self, index: int) -> Optional[ChunkResult]:
        return None

    def put(self, index: int, segments: List[Dict], language: str, text: str) -> None:
        return None

    def count(self) -> int:
        return 0

    def clear(self) -> None:
        return None
