"""Call an OpenAI-compatible /audio/transcriptions endpoint."""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

from .config import Settings
from .errors import ModelError

logger = logging.getLogger(__name__)

# 8 attempts in total, the tail waits up to 10 min for stubborn rate limits.
RETRY_BACKOFFS = [5, 15, 60, 180, 300, 420, 600]


CONNECT_TIMEOUT_SECONDS = 10.0


def _timeout(read_seconds: float):
    """A short connect timeout with a long read timeout, uploads may take a while.

    openai re-exports the Timeout class of the HTTP library it ships with, so
    the right class comes from openai itself instead of a guessed module name.
    A plain number would set all four fields at once and stretch the connect
    guard to the read timeout.
    """
    from openai import Timeout

    return Timeout(
        connect=CONNECT_TIMEOUT_SECONDS,
        read=read_seconds,
        write=CONNECT_TIMEOUT_SECONDS,
        pool=CONNECT_TIMEOUT_SECONDS,
    )


def make_client(settings: Settings):
    from openai import OpenAI

    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=_timeout(settings.timeout),
    )


def is_transient_error(e: Exception) -> bool:
    """Match transient upstream failures worth retrying (5xx, 429, timeouts, connection errors)."""
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(status, int) and (500 <= status < 600 or status == 429):
        return True
    msg = str(e).lower()
    return any(
        kw in msg
        for kw in (
            "internal server error", "bad gateway", "service unavailable",
            "gateway timeout", "timeout", "connection", "temporarily",
            "rate limit", "429",
        )
    )


def _parse(result) -> Tuple[List[Dict], str, str]:
    """Segments, language and text from a verbose_json answer (object, dict or string)."""
    if hasattr(result, "segments") and result.segments:
        segs = [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in result.segments
            if (getattr(s, "text", "") or "").strip()
        ]
        return segs, getattr(result, "language", "") or "", (getattr(result, "text", "") or "").strip()
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except ValueError:
            return [], "", result.strip()
    elif isinstance(result, dict):
        data = result
    elif hasattr(result, "model_dump"):
        data = result.model_dump()
    else:
        return [], "", ""
    if not isinstance(data, dict):
        return [], "", str(data).strip()
    segs = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
        for s in (data.get("segments") or [])
        if (s.get("text") or "").strip()
    ]
    return segs, data.get("language", "") or "", (data.get("text") or "").strip()


def transcribe_verbose(client, settings: Settings, file_path: str) -> Tuple[List[Dict], str, str]:
    """Transcribe one file in its source language. Returns (segments, language, text)."""
    extra = {"language": settings.language} if settings.language else {}
    last_err: Optional[Exception] = None
    for attempt in range(len(RETRY_BACKOFFS) + 1):
        try:
            with open(file_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model=settings.model,
                    file=f,
                    response_format="verbose_json",
                    **extra,
                )
            return _parse(result)
        except Exception as e:
            last_err = e
            if attempt < len(RETRY_BACKOFFS) and is_transient_error(e):
                wait = RETRY_BACKOFFS[attempt]
                logger.warning(
                    "Transcription error (attempt %d/%d): %s, retrying in %ss",
                    attempt + 1, len(RETRY_BACKOFFS) + 1, e, wait,
                )
                time.sleep(wait)
                continue
            break
    raise ModelError(f"Transcription request to {settings.base_url} failed: {last_err}") from last_err
