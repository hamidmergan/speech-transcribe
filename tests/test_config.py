"""Settings come from the argument, the environment or the default, in that order."""

import pytest

from speech_transcribe import config
from speech_transcribe.errors import ConfigError

ENV = config.ENV_PREFIX


def test_base_url_and_model_have_no_default():
    with pytest.raises(ConfigError) as e:
        config.resolve()
    assert "base_url" in str(e.value) and "model" in str(e.value)


def test_the_error_names_the_flag_and_the_variable():
    with pytest.raises(ConfigError) as e:
        config.resolve(model="m")
    message = str(e.value)
    assert "--base-url" in message
    assert ENV + "BASE_URL" in message
    assert "--model" not in message  # only the missing setting is named


def test_a_missing_model_is_named_on_its_own():
    with pytest.raises(ConfigError) as e:
        config.resolve(base_url="http://localhost:11434/v1")
    assert "--model" in str(e.value)


def test_settings_from_the_environment(monkeypatch):
    monkeypatch.setenv(ENV + "BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv(ENV + "MODEL", "your-model")
    monkeypatch.setenv(ENV + "LANGUAGE", "de")
    settings = config.resolve()
    assert settings.base_url == "http://localhost:11434/v1"
    assert settings.model == "your-model"
    assert settings.language == "de"


def test_the_argument_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv(ENV + "MODEL", "from-env")
    settings = config.resolve(base_url="http://localhost:11434/v1", model="from-argument")
    assert settings.model == "from-argument"


def test_an_empty_variable_counts_as_unset(monkeypatch):
    monkeypatch.setenv(ENV + "BASE_URL", "   ")
    with pytest.raises(ConfigError):
        config.resolve(model="m")


def test_without_a_key_the_placeholder_is_used():
    """Local servers need no key, the client library still wants a value."""
    settings = config.resolve(base_url="http://localhost:11434/v1", model="m")
    assert settings.api_key == config.PLACEHOLDER_API_KEY


def test_the_openai_key_is_picked_up(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai")
    settings = config.resolve(base_url="http://localhost:11434/v1", model="m")
    assert settings.api_key == "sk-from-openai"


def test_the_own_key_wins_over_the_openai_one(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai")
    monkeypatch.setenv(ENV + "API_KEY", "sk-own")
    settings = config.resolve(base_url="http://localhost:11434/v1", model="m")
    assert settings.api_key == "sk-own"


def test_the_defaults():
    settings = config.resolve(base_url="http://localhost:11434/v1", model="m")
    assert settings.chunk_seconds == 540
    assert settings.max_file_mb == 500
    assert settings.max_duration_min == 100
    assert settings.concurrency == 2
    assert settings.timeout == 300
    assert settings.language is None


def test_numbers_from_the_environment_are_converted(monkeypatch):
    monkeypatch.setenv(ENV + "CHUNK_SECONDS", "120")
    monkeypatch.setenv(ENV + "CONCURRENCY", "4")
    settings = config.resolve(base_url="http://localhost:11434/v1", model="m")
    assert settings.chunk_seconds == 120.0
    assert settings.concurrency == 4


def test_a_number_that_is_no_number_is_reported(monkeypatch):
    monkeypatch.setenv(ENV + "CONCURRENCY", "many")
    with pytest.raises(ConfigError) as e:
        config.resolve(base_url="http://localhost:11434/v1", model="m")
    assert ENV + "CONCURRENCY" in str(e.value)


@pytest.mark.parametrize(
    "name,value",
    [("chunk_seconds", 0), ("max_file_mb", 0), ("max_duration_min", -1), ("concurrency", 0), ("timeout", 0)],
)
def test_values_must_be_positive(name, value):
    with pytest.raises(ConfigError, match="greater than 0"):
        config.resolve(base_url="http://localhost:11434/v1", model="m", **{name: value})


def test_a_chunk_shorter_than_half_a_minute_is_refused():
    with pytest.raises(ConfigError, match="at least 30"):
        config.resolve(base_url="http://localhost:11434/v1", model="m", chunk_seconds=10)


def test_the_work_folder_sits_in_the_cache_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV + "WORK_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    settings = config.resolve(base_url="http://localhost:11434/v1", model="m")
    assert settings.work_dir == tmp_path / "speech-transcribe"
