"""The web page and its JSON API, against the fake server."""

import json
import time

import pytest
from conftest import needs_ffmpeg

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from speech_transcribe.web.app import create_app  # noqa: E402


@pytest.fixture
def client(server, tmp_path):
    app = create_app(
        allowed_hosts=["testserver"],
        base_url=server.base_url,
        model="speech-model",
        work_dir=str(tmp_path / "work"),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def blank_client(tmp_path):
    """A server started without base_url and model, the page has to supply them."""
    app = create_app(work_dir=str(tmp_path / "work"))
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, path, **fields):
    with open(path, "rb") as f:
        return client.post(
            "/api/transcribe", files={"file": (path.name, f, "audio/mpeg")}, data=fields
        )


def _wait(client, seconds=30):
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = client.get("/api/job").json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("the transcription did not finish")


# --- the page itself --------------------------------------------------------


def test_the_page_is_served(client):
    answer = client.get("/")
    assert answer.status_code == 200
    assert "Audio to text" in answer.text
    assert "Start transcription" in answer.text


def test_the_stylesheet_and_the_script_are_served(client):
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_the_page_offers_every_format(client):
    page = client.get("/").text
    for value in ("txt", "srt", "vtt", "json"):
        assert f'value="{value}"' in page


def test_info_names_the_model_and_the_limits(client, server):
    info = client.get("/api/info").json()
    assert info["configured"] is True
    assert info["model"] == "speech-model"
    assert info["base_url"] == server.base_url
    assert info["max_file_mb"] == 500
    assert info["max_duration_min"] == 100
    assert ".mp3" in info["extensions"] and ".mp4" in info["extensions"]
    assert info["formats"] == ["txt", "vtt", "srt", "json"]


def test_info_says_when_nothing_is_set(blank_client):
    info = blank_client.get("/api/info").json()
    assert info["configured"] is False
    assert info["model"] == ""


# --- the guard --------------------------------------------------------------


def test_an_unknown_host_is_refused(client):
    assert client.get("/", headers={"host": "evil.example"}).status_code == 400


def test_another_origin_is_refused(client):
    answer = client.get("/api/info", headers={"origin": "http://evil.example"})
    assert answer.status_code == 403


def test_the_own_origin_passes(client):
    assert client.get("/api/info", headers={"origin": "http://testserver"}).status_code == 200


def test_an_upload_over_the_limit_is_refused_before_it_is_read(server, tmp_path):
    app = create_app(base_url=server.base_url, model="m", max_file_mb=1, work_dir=str(tmp_path))
    with TestClient(app) as small:
        answer = small.post(
            "/api/transcribe",
            files={"file": ("big.mp3", b"\x00" * 3 * 1024 * 1024, "audio/mpeg")},
        )
    assert answer.status_code == 413
    assert "max_file_mb" in answer.text


# --- transcribing -----------------------------------------------------------


@needs_ffmpeg
def test_a_recording_becomes_a_transcript(client, copy_media, server):
    assert _upload(client, copy_media("short")).status_code == 200
    job = _wait(client)
    assert job["state"] == "done"
    assert job["text"] == "request 1 second"
    assert job["filename"] == "short.txt"
    assert job["language"] == "english"
    assert job["chunks"] == 1
    assert len(server.requests) == 1


@needs_ffmpeg
@pytest.mark.parametrize(
    "fmt,suffix,marker",
    [("vtt", ".vtt", "WEBVTT"), ("srt", ".srt", "00:00:01,000"), ("json", ".json", '"segments"')],
)
def test_every_format_comes_back(client, copy_media, fmt, suffix, marker):
    _upload(client, copy_media("short"), response_format=fmt)
    job = _wait(client)
    assert job["filename"] == "short" + suffix
    assert marker in job["text"]


@needs_ffmpeg
def test_the_transcript_can_be_downloaded(client, copy_media):
    _upload(client, copy_media("short"), response_format="vtt")
    _wait(client)
    answer = client.get("/api/result")
    assert answer.status_code == 200
    assert answer.text.startswith("WEBVTT")
    assert "short.vtt" in answer.headers["content-disposition"]


def test_there_is_nothing_to_download_yet(client):
    assert client.get("/api/result").status_code == 409


@needs_ffmpeg
def test_the_language_from_the_page_reaches_the_server(client, copy_media, server):
    _upload(client, copy_media("short"), language="de")
    _wait(client)
    assert server.requests[0]["language"] == "de"


@needs_ffmpeg
def test_short_cues_shorten_the_lines(client, copy_media):
    _upload(client, copy_media("short"), response_format="vtt", short_cues="true")
    job = _wait(client)
    assert all(len(line) <= 42 for line in job["text"].split("\n"))


@needs_ffmpeg
def test_a_long_recording_is_split_and_reported(server, copy_media, tmp_path):
    app = create_app(
        base_url=server.base_url, model="speech-model", chunk_seconds=30, work_dir=str(tmp_path)
    )
    with TestClient(app) as slow:
        _upload(slow, copy_media("long", "long.wav"), response_format="json")
        job = _wait(slow)
    assert job["state"] == "done"
    assert job["chunks"] == 3
    assert job["total"] == 3
    assert len(json.loads(job["text"])["segments"]) == 6


@needs_ffmpeg
def test_the_settings_can_come_from_the_page(blank_client, copy_media, server):
    answer = _upload(
        blank_client, copy_media("short"), base_url=server.base_url, model="from-the-page"
    )
    assert answer.status_code == 200
    job = _wait(blank_client)
    assert job["state"] == "done"
    assert server.requests[0]["model"] == "from-the-page"


# --- refusals ---------------------------------------------------------------


def test_without_settings_the_upload_is_refused(blank_client, tmp_path):
    path = tmp_path / "talk.mp3"
    path.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")
    answer = _upload(blank_client, path)
    assert answer.status_code == 400
    assert "server address" in answer.json()["detail"]


def test_an_unknown_file_type_is_refused(client, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    answer = _upload(client, path)
    assert answer.status_code == 400
    assert ".mp3" in answer.json()["detail"]


def test_an_unknown_format_is_refused(client, tmp_path):
    path = tmp_path / "talk.mp3"
    path.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")
    answer = _upload(client, path, response_format="docx")
    assert answer.status_code == 400
    assert "Invalid format" in answer.json()["detail"]


@needs_ffmpeg
def test_a_refused_file_lands_in_the_job_error(client, copy_media, tmp_path):
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"this is not audio at all" * 10)
    assert _upload(client, broken).status_code == 200
    job = _wait(client)
    assert job["state"] == "error"
    assert "does not match the extension" in job["error"]


@needs_ffmpeg
def test_a_server_error_lands_in_the_job_error(client, copy_media, server):
    server.fail_on = {1}
    _upload(client, copy_media("short"))
    job = _wait(client)
    assert job["state"] == "error"
    assert job["error"]


def test_an_unknown_setting_is_refused(tmp_path):
    from speech_transcribe.errors import ConfigError

    with pytest.raises(ConfigError, match="Unknown setting"):
        create_app(colour="blue")


def test_the_environment_configures_the_page(monkeypatch, server, tmp_path):
    monkeypatch.setenv("SPEECH_TRANSCRIBE_BASE_URL", server.base_url)
    monkeypatch.setenv("SPEECH_TRANSCRIBE_MODEL", "from-the-environment")
    app = create_app(work_dir=str(tmp_path / "work"))
    with TestClient(app) as from_env:
        info = from_env.get("/api/info").json()
    assert info["configured"] is True
    assert info["model"] == "from-the-environment"
