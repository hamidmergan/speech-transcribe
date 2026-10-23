import json
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not available")


def _field(body: bytes, name: str):
    m = re.search(rb'name="' + name.encode() + rb'"\r\n\r\n(.*?)\r\n--', body, re.S)
    return m.group(1).decode() if m else None


class FakeServer:
    """OpenAI-compatible /audio/transcriptions that answers with verbose_json.

    Every answer has two segments at 1.0-2.5 s and 5.0-6.0 s of the uploaded
    chunk. The first text names the request number, so the merged order shows.
    """

    def __init__(self):
        self.requests = []
        self.fail_on = set()  # request numbers (1-based) that fail with 400
        self.status_on_fail = 400
        self.no_segments = False
        self.empty_on = set()  # request numbers that answer 200 with nothing in it
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                with outer.lock:
                    outer.requests.append(
                        {
                            "path": self.path,
                            "headers": {k.lower(): v for k, v in self.headers.items()},
                            "model": _field(body, "model"),
                            "language": _field(body, "language"),
                            "response_format": _field(body, "response_format"),
                            "size": len(body),
                        }
                    )
                    n = len(outer.requests)
                if n in outer.fail_on:
                    data = json.dumps({"error": {"message": "bad chunk"}}).encode()
                    self.send_response(outer.status_on_fail)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                language = outer.requests[-1]["language"] or "english"
                segments = [
                    {"id": 0, "start": 1.0, "end": 2.5, "text": f" request {n} "},
                    {"id": 1, "start": 5.0, "end": 6.0, "text": " second "},
                    {"id": 2, "start": 6.0, "end": 6.5, "text": "   "},
                ]
                empty = n in outer.empty_on
                payload = {
                    "task": "transcribe",
                    "language": language,
                    "duration": 7.0,
                    "text": "" if empty else f"request {n} second",
                    "segments": [] if empty or outer.no_segments else segments,
                }
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.stop()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    import os

    for name in list(os.environ):
        if name.startswith("SPEECH_TRANSCRIBE_") or name == "OPENAI_API_KEY":
            monkeypatch.delenv(name)
    # never touch the real cache folder
    monkeypatch.setenv("SPEECH_TRANSCRIBE_WORK_DIR", str(tmp_path / "work"))


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


# 8 s tone, 2 s pause, repeated: pauses start at 8, 18, 28, 38, ...
_PATTERN = "aevalsrc=if(lt(mod(t\\,10)\\,8)\\,0.5*sin(2*PI*440*t)\\,0):s=16000:d={d}"


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    """Real media files made with ffmpeg: a 75 s WAV with pauses, a short MP3 and a short MP4."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not available")
    folder = tmp_path_factory.mktemp("media")
    long_wav = folder / "long.wav"
    _ffmpeg("-f", "lavfi", "-i", _PATTERN.format(d=75), str(long_wav))
    short_mp3 = folder / "short.mp3"
    _ffmpeg("-f", "lavfi", "-i", _PATTERN.format(d=12), "-c:a", "libmp3lame", str(short_mp3))
    video = folder / "clip.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=12",
        "-f", "lavfi", "-i", _PATTERN.format(d=12),
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(video),
    )
    return {"long": long_wav, "short": short_mp3, "video": video}


@pytest.fixture
def copy_media(media, tmp_path):
    """A fresh copy of one media file in the test folder, so outputs land there."""

    def _copy(key: str, name=None):
        target = tmp_path / (name or media[key].name)
        shutil.copy(media[key], target)
        return target

    return _copy
