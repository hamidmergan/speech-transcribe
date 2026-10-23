"""The command line against the fake server."""

import json

import pytest
from conftest import needs_ffmpeg

from speech_transcribe.cli import main


def _args(server, *extra):
    return ["--base-url", server.base_url, "--model", "speech-model", "--quiet", *extra]


# --- configuration and help -------------------------------------------------


def test_without_settings_the_cli_says_what_is_missing(capsys, tmp_path):
    code = main([str(tmp_path / "talk.mp3")])
    assert code == 2
    err = capsys.readouterr().err
    assert "--base-url" in err and "--model" in err


def test_the_help_is_english(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for word in ("Transcribe audio and video", "Output format", "--resume", "--language"):
        assert word in out


def test_the_version_is_printed(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "speech-transcribe" in capsys.readouterr().out


def test_stdout_and_output_file_cannot_be_combined(capsys, tmp_path, server):
    code = main(_args(server, "--stdout", "-o", str(tmp_path / "x.txt"), "a.mp3"))
    assert code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_an_unknown_format_is_refused(capsys, server):
    with pytest.raises(SystemExit):
        main(_args(server, "--format", "docx", "a.mp3"))
    assert "invalid choice" in capsys.readouterr().err


def test_a_missing_file_is_reported(capsys, server, tmp_path):
    code = main(_args(server, str(tmp_path / "gone.mp3")))
    assert code == 1
    assert "File not found" in capsys.readouterr().err


# --- writing results --------------------------------------------------------


@needs_ffmpeg
def test_the_result_lands_next_to_the_input(server, copy_media):
    audio = copy_media("short")
    assert main(_args(server, str(audio))) == 0
    written = audio.with_suffix(".txt").read_text(encoding="utf-8")
    assert written.strip() == "request 1 second"
    assert len(server.requests) == 1


@needs_ffmpeg
@pytest.mark.parametrize("fmt,marker", [("vtt", "WEBVTT"), ("srt", "00:00:01,000 -->")])
def test_subtitle_formats_are_written(server, copy_media, fmt, marker):
    audio = copy_media("short")
    assert main(_args(server, "--format", fmt, str(audio))) == 0
    assert marker in audio.with_suffix("." + fmt).read_text(encoding="utf-8")


@needs_ffmpeg
def test_json_holds_the_segments(server, copy_media):
    audio = copy_media("short")
    assert main(_args(server, "--format", "json", str(audio))) == 0
    data = json.loads(audio.with_suffix(".json").read_text(encoding="utf-8"))
    assert [s["start"] for s in data["segments"]] == [1.0, 5.0]


@needs_ffmpeg
def test_an_output_file_can_be_named(server, copy_media, tmp_path):
    audio = copy_media("short")
    target = tmp_path / "out" / "talk.vtt"
    assert main(_args(server, "--format", "vtt", "-o", str(target), str(audio))) == 0
    assert target.read_text(encoding="utf-8").startswith("WEBVTT")


@needs_ffmpeg
def test_stdout_prints_instead_of_writing(server, copy_media, capsys):
    audio = copy_media("short")
    assert main(_args(server, "--stdout", str(audio))) == 0
    assert "request 1 second" in capsys.readouterr().out
    assert not audio.with_suffix(".txt").exists()


@needs_ffmpeg
def test_an_existing_result_is_not_overwritten(server, copy_media):
    audio = copy_media("short")
    first = audio.with_suffix(".txt")
    first.write_text("older run\n", encoding="utf-8")
    assert main(_args(server, str(audio))) == 0
    assert first.read_text(encoding="utf-8") == "older run\n"
    assert (audio.parent / "short (1).txt").exists()


@needs_ffmpeg
def test_several_files_go_into_one_folder(server, copy_media, tmp_path):
    first = copy_media("short", "one.mp3")
    second = copy_media("short", "two.mp3")
    out = tmp_path / "results"
    assert main(_args(server, "-o", str(out), str(first), str(second))) == 0
    assert sorted(p.name for p in out.iterdir()) == ["one.txt", "two.txt"]


@needs_ffmpeg
def test_several_files_with_the_same_stem_keep_their_extension(server, copy_media, tmp_path):
    first = copy_media("short", "talk.mp3")
    second = copy_media("video", "talk.mp4")
    out = tmp_path / "results"
    assert main(_args(server, "-o", str(out), str(first), str(second))) == 0
    assert sorted(p.name for p in out.iterdir()) == ["talk.mp3.txt", "talk.mp4.txt"]


@needs_ffmpeg
def test_one_broken_file_does_not_stop_the_others(server, copy_media, tmp_path, capsys):
    good = copy_media("short")
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"this is not audio at all" * 10)
    code = main(_args(server, str(good), str(broken)))
    assert code == 1
    assert good.with_suffix(".txt").exists()
    assert "does not match the extension" in capsys.readouterr().err


@needs_ffmpeg
def test_the_language_reaches_the_server(server, copy_media):
    audio = copy_media("short")
    assert main(_args(server, "--language", "de", str(audio))) == 0
    assert server.requests[0]["language"] == "de"


@needs_ffmpeg
def test_short_cues_shorten_the_lines(server, copy_media):
    audio = copy_media("short")
    assert main(_args(server, "--format", "vtt", "--short-cues", str(audio))) == 0
    text = audio.with_suffix(".vtt").read_text(encoding="utf-8")
    assert all(len(line) <= 42 for line in text.split("\n"))


@needs_ffmpeg
def test_progress_is_printed_unless_quiet(server, copy_media, capsys, tmp_path):
    audio = copy_media("long", "long.wav")
    code = main([
        "--base-url", server.base_url, "--model", "speech-model",
        "--chunk-seconds", "30", str(audio),
    ])
    assert code == 0
    err = capsys.readouterr().err
    assert "chunk 1/3" in err and "chunk 3/3" in err
    assert "Wrote " in err


@needs_ffmpeg
def test_a_long_recording_is_split_and_merged(server, copy_media):
    audio = copy_media("long", "long.wav")
    code = main(_args(server, "--chunk-seconds", "30", "--format", "json", str(audio)))
    assert code == 0
    data = json.loads(audio.with_suffix(".json").read_text(encoding="utf-8"))
    assert len(server.requests) == 3
    assert len(data["segments"]) == 6
    assert data["segments"][0]["start"] == 1.0
    assert data["segments"][2]["start"] > 20  # the second chunk is shifted


@needs_ffmpeg
def test_a_failed_chunk_points_at_resume(server, copy_media, capsys):
    audio = copy_media("long", "long.wav")
    server.fail_on = {2}
    code = main(_args(server, "--chunk-seconds", "30", "--concurrency", "1", str(audio)))
    assert code == 1
    err = capsys.readouterr().err
    assert "Chunk 2 transcription failed" in err
    assert "--resume" in err


@needs_ffmpeg
def test_resume_finishes_the_interrupted_run(server, copy_media):
    audio = copy_media("long", "long.wav")
    server.fail_on = {2}
    assert main(_args(server, "--chunk-seconds", "30", "--concurrency", "1", str(audio))) == 1
    sent = len(server.requests)
    server.fail_on = set()

    assert main(_args(server, "--chunk-seconds", "30", "--concurrency", "1", "--resume", str(audio))) == 0

    assert len(server.requests) == sent + 1
    assert audio.with_suffix(".txt").exists()


# --- nothing escapes as a traceback -----------------------------------------


@needs_ffmpeg
def test_an_unexpected_error_is_printed_as_a_message(server, copy_media, capsys, monkeypatch):
    """A bare ValueError out of the pipeline used to reach the terminal as a traceback."""
    from speech_transcribe import pipeline

    def boom(path):
        raise ValueError("could not convert string to float: 'N/A'")

    monkeypatch.setattr(pipeline, "get_audio_duration", boom)
    code = main(_args(server, "--stdout", str(copy_media("short"))))
    assert code == 1
    captured = capsys.readouterr()
    assert "Error for" in captured.err
    assert "Traceback" not in captured.err and "Traceback" not in captured.out


@needs_ffmpeg
def test_an_unwritable_output_is_reported(server, copy_media, capsys, tmp_path):
    """-o pointing at an existing folder used to crash after the model was paid for."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "short.txt").mkdir()
    code = main(_args(server, "-o", str(blocked / "short.txt"), str(copy_media("short"))))
    assert code == 1
    assert "Could not write" in capsys.readouterr().err
