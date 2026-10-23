"""What the package exports."""

import speech_transcribe


def test_the_mp4_helpers_are_reachable_through_the_package():
    for name in (
        "is_fragmented_mp4",
        "is_fragmented_mp4_reader",
        "normalize_if_fragmented",
        "normalize_for_playback",
    ):
        assert name in speech_transcribe.__all__
        assert callable(getattr(speech_transcribe, name))


def test_everything_in_all_exists():
    for name in speech_transcribe.__all__:
        assert hasattr(speech_transcribe, name), name


def test_the_build_context_leaves_the_heavy_folders_out():
    """COPY . . would otherwise put .venv, .git and private files into the image."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    ignored = (root / ".dockerignore").read_text(encoding="utf-8").split()
    for entry in (".git", ".venv", ".pytest_cache", "__pycache__", "private/", ".*/"):
        assert entry in ignored
