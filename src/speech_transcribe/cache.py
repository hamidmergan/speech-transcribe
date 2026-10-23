"""Remembers the result per chunk on disk, so a second attempt repeats nothing."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ChunkResult = Tuple[List[Dict], str, str]  # segments, language, text


def cache_key(path, model: str, language: Optional[str], chunk_seconds: float) -> str:
    """Key from the file content and everything that changes the result."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(f"\0{model}\0{language or ''}\0{chunk_seconds:g}".encode("utf-8"))
    return digest.hexdigest()[:32]


class ChunkCache:
    """One folder per recording, one JSON file per finished chunk. Best effort."""

    def __init__(self, work_dir, key: str):
        self.dir = Path(work_dir) / key

    def _file(self, index: int) -> Path:
        return self.dir / f"chunk_{index:03d}.json"

    def get(self, index: int) -> Optional[ChunkResult]:
        """Return the cached (segments, language, text) for a chunk, or None."""
        try:
            data = json.loads(self._file(index).read_text(encoding="utf-8"))
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
