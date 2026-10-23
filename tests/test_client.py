"""The request to the speech-to-text server: retries and reading the answer."""

from types import SimpleNamespace

import pytest

from speech_transcribe import client as stt
from speech_transcribe import config
from speech_transcribe.errors import ModelError


class _Stub:
    """Stands in for the OpenAI client, answers or raises per attempt."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []
        create = self._create
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class _Status(Exception):
    def __init__(self, status_code, message="upstream said no"):
        super().__init__(f"Error code: {status_code} - {message}")
        self.status_code = status_code


@pytest.fixture
def settings(tmp_path):
    return config.resolve(base_url="http://localhost:1/v1", model="speech-model", work_dir=str(tmp_path))


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "chunk.mp3"
    path.write_bytes(b"audio")
    return str(path)


@pytest.fixture
def no_waiting(monkeypatch):
    waits = []
    monkeypatch.setattr(stt.time, "sleep", lambda s: waits.append(s))
    return waits


_ANSWER = {
    "language": "german",
    "text": "Good morning",
    "segments": [{"start": 0.0, "end": 1.0, "text": " Good morning "}],
}


# --- what counts as temporary ----------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_server_side_status_codes_are_temporary(status):
    assert stt.is_transient_error(_Status(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_side_status_codes_are_permanent(status):
    assert stt.is_transient_error(_Status(status)) is False


@pytest.mark.parametrize(
    "message",
    ["Read timeout", "Connection refused", "Service Unavailable", "rate limit reached"],
)
def test_known_wordings_are_temporary(message):
    assert stt.is_transient_error(RuntimeError(message)) is True


def test_an_unknown_error_is_permanent():
    assert stt.is_transient_error(ValueError("file is not audio")) is False


# --- retrying ---------------------------------------------------------------


def test_a_temporary_error_is_retried_with_the_fixed_backoffs(settings, audio, no_waiting):
    client = _Stub([_Status(503), _Status(503), _ANSWER])
    segments, language, text = stt.transcribe_verbose(client, settings, audio)
    assert no_waiting == stt.RETRY_BACKOFFS[:2]
    assert text == "Good morning"
    assert language == "german"
    assert len(client.calls) == 3


def test_a_permanent_error_fails_at_once(settings, audio, no_waiting):
    client = _Stub([_Status(400)])
    with pytest.raises(ModelError) as e:
        stt.transcribe_verbose(client, settings, audio)
    assert no_waiting == []
    assert len(client.calls) == 1
    assert settings.base_url in str(e.value)


def test_the_retries_run_out(settings, audio, no_waiting):
    attempts = len(stt.RETRY_BACKOFFS) + 1
    client = _Stub([_Status(503) for _ in range(attempts)])
    with pytest.raises(ModelError):
        stt.transcribe_verbose(client, settings, audio)
    assert len(client.calls) == attempts
    assert no_waiting == stt.RETRY_BACKOFFS


def test_the_backoffs_grow_and_end_at_ten_minutes():
    assert stt.RETRY_BACKOFFS == sorted(stt.RETRY_BACKOFFS)
    assert stt.RETRY_BACKOFFS[-1] == 600


# --- reading the answer -----------------------------------------------------


def test_verbose_json_is_requested_in_the_source_language(settings, audio):
    client = _Stub([_ANSWER])
    stt.transcribe_verbose(client, settings, audio)
    assert client.calls[0]["response_format"] == "verbose_json"
    assert client.calls[0]["model"] == "speech-model"
    assert "language" not in client.calls[0]  # no language set, the server detects it


def test_a_set_language_is_sent_along(tmp_path, audio):
    settings = config.resolve(
        base_url="http://localhost:1/v1", model="m", language="de", work_dir=str(tmp_path)
    )
    client = _Stub([_ANSWER])
    stt.transcribe_verbose(client, settings, audio)
    assert client.calls[0]["language"] == "de"


def test_segment_text_is_trimmed(settings, audio):
    segments, _, _ = stt.transcribe_verbose(client=_Stub([_ANSWER]), settings=settings, file_path=audio)
    assert segments == [{"start": 0.0, "end": 1.0, "text": "Good morning"}]


def test_an_answer_object_is_read_like_a_dict(settings, audio):
    answer = SimpleNamespace(
        language="german",
        text="Good morning",
        segments=[SimpleNamespace(start=0.0, end=1.0, text=" Good morning ")],
    )
    segments, language, text = stt.transcribe_verbose(_Stub([answer]), settings, audio)
    assert segments == [{"start": 0.0, "end": 1.0, "text": "Good morning"}]
    assert (language, text) == ("german", "Good morning")


def test_a_json_string_answer_is_read_too(settings, audio):
    import json

    segments, language, _ = stt.transcribe_verbose(_Stub([json.dumps(_ANSWER)]), settings, audio)
    assert segments[0]["text"] == "Good morning"
    assert language == "german"


def test_a_plain_text_answer_keeps_its_text(settings, audio):
    segments, language, text = stt.transcribe_verbose(_Stub(["Good morning"]), settings, audio)
    assert (segments, language, text) == ([], "", "Good morning")


def test_a_timeout_is_set_on_the_client(settings):
    timeout = stt._timeout(settings.timeout)
    assert getattr(timeout, "read", timeout) == settings.timeout
