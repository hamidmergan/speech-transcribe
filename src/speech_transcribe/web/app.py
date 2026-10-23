"""Web API for the transcription page.

Upload a recording, pick the output format, watch the parts come in, read and
download the transcript. The upload is deleted as soon as it is transcribed.
"""

import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..config import CHUNK_SECONDS, MAX_DURATION_MIN, MAX_FILE_MB, resolve
from ..core import EXTENSIONS, FORMATS, Transcript, normalize_format, transcribe
from ..errors import ConfigError, TranscribeError
from ..validation import EXTENSIONS as MEDIA_EXTENSIONS

STATIC = Path(__file__).parent / "static"
MB = 1024 * 1024

# Everything create_app may be given, straight through to transcribe().
OPTION_NAMES = (
    "base_url", "model", "api_key", "language", "chunk_seconds",
    "max_file_mb", "max_duration_min", "concurrency", "timeout", "work_dir",
)


class Job:
    """The one transcription the page is working on."""

    def __init__(self) -> None:
        self.state = "idle"  # idle, running, done or error
        self.current = 0
        self.total = 0
        self.error = ""
        self.name = ""  # name of the uploaded file
        self.format = "txt"
        self.text = ""
        self.transcript: Optional[Transcript] = None

    def reset(self, name: str, fmt: str) -> None:
        self.state, self.current, self.total, self.error = "running", 0, 0, ""
        self.name, self.format = name, fmt
        self.text, self.transcript = "", None

    def filename(self) -> str:
        return Path(self.name).stem + EXTENSIONS[self.format]


def _hostname(host: str) -> str:
    if host.startswith("["):
        return host[1:host.find("]")].lower()
    return host.rsplit(":", 1)[0].lower() if host.count(":") == 1 else host.lower()


def _safe_name(name: str) -> str:
    """Keep the file name the browser sent, without any path in it."""
    return Path(name.replace("\\", "/")).name or "recording"


def create_app(allowed_hosts: Optional[Sequence[str]] = None, **options) -> FastAPI:
    """The transcription page and its JSON API, with one recording at a time.

    `options` are the defaults for transcribe(). base_url, model and api_key can
    also come from fields on the page, so the server may start without them.
    `allowed_hosts` limits the Host header (against DNS rebinding), requests from
    another origin are refused.
    """
    unknown = set(options) - set(OPTION_NAMES)
    if unknown:
        raise ConfigError(f"Unknown setting: {', '.join(sorted(unknown))}")
    defaults = {k: v for k, v in options.items() if v is not None}
    try:
        # server address, model and key may also come from the environment
        settings = resolve(**defaults)
    except ConfigError:
        pass
    else:
        defaults.update(
            base_url=settings.base_url, model=settings.model, api_key=settings.api_key,
            language=settings.language or "", chunk_seconds=settings.chunk_seconds,
            max_file_mb=settings.max_file_mb, max_duration_min=settings.max_duration_min,
            concurrency=settings.concurrency, timeout=settings.timeout,
            work_dir=str(settings.work_dir),
        )
    tmp = tempfile.TemporaryDirectory(prefix="speech-transcribe-")
    work = Path(tmp.name)
    job = Job()
    lock = threading.Lock()
    max_file_mb = float(defaults.get("max_file_mb") or MAX_FILE_MB)
    max_duration_min = float(defaults.get("max_duration_min") or MAX_DURATION_MIN)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        tmp.cleanup()

    app = FastAPI(title="speech-transcribe", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    hosts = {h.lower() for h in allowed_hosts} if allowed_hosts is not None else None

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        if hosts is not None and _hostname(host) not in hosts:
            return PlainTextResponse("Unknown host", status_code=400)
        origin = request.headers.get("origin")
        own = {host.lower(), request.headers.get("x-forwarded-host", "").lower()}  # also behind a reverse proxy
        if origin is not None and urlsplit(origin).netloc.lower() not in own:
            return PlainTextResponse("Cross-origin requests are not allowed", status_code=403)
        if request.url.path == "/api/transcribe":
            # refuse a too large upload before it is read, one MB of room for the form around the file
            length = request.headers.get("content-length", "")
            if not length.isdigit():
                return PlainTextResponse("The upload needs a Content-Length header.", status_code=411)
            if int(length) > max_file_mb * MB + MB:
                return PlainTextResponse(
                    f"The upload is larger than {max_file_mb:g} MB (max_file_mb).", status_code=413
                )
        return await call_next(request)

    def chosen_options(base_url: str, model: str, api_key: str) -> dict:
        """The settings of this run: the fields on the page first, then the server's."""
        chosen = dict(defaults)
        for name, value in (("base_url", base_url), ("model", model), ("api_key", api_key)):
            if value.strip():
                chosen[name] = value.strip()
        return chosen

    def run(path: Path, fmt: str, language: str, short_cues: bool, chosen: dict) -> None:
        def progress(done: int, total: int) -> None:
            with lock:
                job.current, job.total = done, total

        try:
            result = transcribe(
                path, language=language or None, fmt=fmt, on_progress=progress, **chosen
            )
        except TranscribeError as exc:
            with lock:
                job.state, job.error = "error", str(exc)
            return
        finally:
            path.unlink(missing_ok=True)
        with lock:
            job.state, job.transcript = "done", result
            job.text = result.render(fmt, short_cues=short_cues)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info():
        return {
            "model": defaults.get("model") or "",
            "base_url": defaults.get("base_url") or "",
            "configured": bool(defaults.get("base_url") and defaults.get("model")),
            "language": defaults.get("language") or "",
            "formats": list(FORMATS),
            "extensions": sorted(MEDIA_EXTENSIONS),
            "max_file_mb": max_file_mb,
            "max_duration_min": max_duration_min,
            "chunk_minutes": round(float(defaults.get("chunk_seconds") or CHUNK_SECONDS) / 60, 1),
        }

    @app.post("/api/transcribe")
    def start(
        file: UploadFile = File(...),
        response_format: str = Form("txt"),
        language: str = Form(""),
        short_cues: bool = Form(False),
        base_url: str = Form(""),
        model: str = Form(""),
        api_key: str = Form(""),
    ):
        try:
            fmt = normalize_format(response_format)
        except ConfigError as exc:
            raise HTTPException(400, str(exc))
        with lock:
            if job.state == "running":
                raise HTTPException(409, "A recording is being transcribed. Wait until it is done.")
        name = _safe_name(file.filename or "recording")
        if Path(name).suffix.lower() not in MEDIA_EXTENSIONS:
            raise HTTPException(400, f"Only {', '.join(sorted(MEDIA_EXTENSIONS))} allowed")
        chosen = chosen_options(base_url, model, api_key)
        if not (chosen.get("base_url") and chosen.get("model")):
            raise HTTPException(400, "Set the server address and the model first.")
        spoken = language.strip() or str(chosen.pop("language", "") or "")
        chosen.pop("language", None)
        target = work / name
        with open(target, "wb") as out:
            shutil.copyfileobj(file.file, out)
        with lock:
            job.reset(name, fmt)
        threading.Thread(
            target=run, args=(target, fmt, spoken, short_cues, chosen), daemon=True
        ).start()
        return {"state": "running", "name": name}

    @app.get("/api/job")
    def state():
        with lock:
            answer = {
                "state": job.state,
                "current": job.current,
                "total": job.total,
                "error": job.error,
                "name": job.name,
                "format": job.format,
            }
            if job.state == "done" and job.transcript is not None:
                answer.update(
                    filename=job.filename(),
                    language=job.transcript.language,
                    duration=job.transcript.duration,
                    chunks=job.transcript.chunks,
                    text=job.text,
                )
            return answer

    @app.get("/api/result")
    def result():
        with lock:
            if job.state != "done" or job.transcript is None:
                raise HTTPException(409, "There is no transcript yet.")
            text, filename = job.text, job.filename()
        path = work / filename
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        return FileResponse(path, media_type="text/plain; charset=utf-8", filename=filename)

    return app
