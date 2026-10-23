import pytest

from speech_transcribe import tools
from speech_transcribe.errors import MissingToolError


def test_missing_tools_named(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    with pytest.raises(MissingToolError) as e:
        tools.check_tools()
    assert "ffmpeg" in str(e.value) and "ffprobe" in str(e.value)


def test_present_tools_pass(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/" + name)
    tools.check_tools()
