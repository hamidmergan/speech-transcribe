"""The four output formats, all rendered from the same segments."""

import json

import pytest

from speech_transcribe.core import Transcript, normalize_format, output_name
from speech_transcribe.errors import ConfigError
from speech_transcribe.subtitles import segments_to_srt, segments_to_vtt, vtt_to_segments

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "Good morning everyone."},
    {"start": 2.5, "end": 6.25, "text": "Today we look at the basics."},
    {"start": 3661.5, "end": 3663.0, "text": "And that was it."},
]


def _transcript(fmt="txt"):
    return Transcript(
        text="Good morning everyone. Today we look at the basics. And that was it.",
        segments=[dict(s) for s in SEGMENTS],
        language="english",
        duration=3663.0,
        format=fmt,
    )


# --- VTT --------------------------------------------------------------------


def test_vtt_starts_with_the_header():
    assert _transcript().to_vtt().startswith("WEBVTT\n\n")


def test_vtt_times_use_a_dot():
    assert "00:00:00.000 --> 00:00:02.500" in _transcript().to_vtt()


def test_vtt_counts_the_cues():
    lines = _transcript().to_vtt().split("\n")
    assert lines[2] == "1"
    assert "3" in lines


def test_vtt_carries_hours():
    assert "01:01:01.500 --> 01:01:03.000" in _transcript().to_vtt()


def test_vtt_reads_back_into_the_same_segments():
    assert vtt_to_segments(_transcript().to_vtt()) == SEGMENTS


# --- SRT --------------------------------------------------------------------


def test_srt_times_use_a_comma():
    assert "00:00:00,000 --> 00:00:02,500" in _transcript().to_srt()


def test_srt_has_no_header():
    assert not _transcript().to_srt().startswith("WEBVTT")
    assert _transcript().to_srt().split("\n")[0] == "1"


def test_srt_and_vtt_hold_the_same_text():
    srt_texts = [s["text"] for s in vtt_to_segments(_transcript().to_srt())]
    assert srt_texts == [s["text"] for s in SEGMENTS]


def test_milliseconds_are_rounded_once():
    """1.9996 s becomes 00:00:02.000 and never 00:00:01.1000."""
    out = segments_to_vtt([{"start": 1.9996, "end": 2.0004, "text": "x"}])
    assert "00:00:02.000 --> 00:00:02.000" in out


def test_a_negative_time_cannot_appear():
    assert "00:00:00,000" in segments_to_srt([{"start": -0.4, "end": 1.0, "text": "x"}])


def test_no_segments_still_gives_a_valid_vtt():
    assert segments_to_vtt([]) == "WEBVTT\n"
    assert segments_to_srt([]) == ""


# --- text and JSON ----------------------------------------------------------


def test_text_is_the_plain_transcript():
    assert _transcript().to_text().startswith("Good morning")
    assert "-->" not in _transcript().to_text()


def test_json_holds_text_segments_language_and_duration():
    data = json.loads(_transcript().to_json())
    assert data["language"] == "english"
    assert data["duration"] == 3663.0
    assert data["segments"] == SEGMENTS
    assert data["text"].startswith("Good morning")


def test_json_keeps_non_ascii_readable():
    transcript = Transcript(text="Grüße", segments=[], format="json")
    assert "Grüße" in transcript.to_json()


# --- choosing the format ----------------------------------------------------


@pytest.mark.parametrize("fmt", ["txt", "vtt", "srt", "json"])
def test_render_follows_the_chosen_format(fmt):
    transcript = _transcript(fmt)
    assert transcript.render() == transcript.render(fmt)
    assert str(transcript) == transcript.render(fmt)


def test_render_can_be_overridden():
    assert _transcript("txt").render("vtt").startswith("WEBVTT")


def test_text_is_an_alias_of_txt():
    assert normalize_format("text") == "txt"
    assert normalize_format("TXT") == "txt"
    assert normalize_format(None) == "txt"


def test_an_unknown_format_is_refused():
    with pytest.raises(ConfigError, match="Invalid format"):
        normalize_format("docx")


@pytest.mark.parametrize(
    "fmt,expected",
    [("txt", "talk.txt"), ("vtt", "talk.vtt"), ("srt", "talk.srt"), ("json", "talk.json")],
)
def test_the_output_name_follows_the_input(fmt, expected):
    assert output_name("/recordings/talk.mp4", fmt) == expected


# --- short cues -------------------------------------------------------------


_LONG = (
    "At scientific conferences there are several people in charge, and depending on "
    "the size of the conference there are several leads, so-called chairs, who are "
    "experts from the respective field."
)


def test_short_cues_split_a_long_segment():
    transcript = Transcript(text=_LONG, segments=[{"start": 0.0, "end": 12.0, "text": _LONG}])
    assert len(vtt_to_segments(transcript.to_vtt(short_cues=True))) > 1
    assert len(vtt_to_segments(transcript.to_vtt())) == 1


def test_short_cues_keep_the_total_time():
    transcript = Transcript(text=_LONG, segments=[{"start": 4.0, "end": 12.0, "text": _LONG}])
    cues = vtt_to_segments(transcript.to_srt(short_cues=True))
    assert cues[0]["start"] == 4.0
    assert cues[-1]["end"] == 12.0


def test_short_cues_do_not_change_the_plain_text():
    transcript = Transcript(text=_LONG, segments=[{"start": 0.0, "end": 12.0, "text": _LONG}])
    assert transcript.render("txt") == _LONG


# --- what may stand in a cue ------------------------------------------------


def test_vtt_escapes_the_markup_characters():
    """A player reads <b> as a tag and &T as a broken entity."""
    vtt = segments_to_vtt([{"start": 0.0, "end": 2.0, "text": "R&D at AT&T <b>grew</b>"}])
    assert "R&amp;D at AT&amp;T &lt;b&gt;grew&lt;/b&gt;" in vtt
    assert "<b>" not in vtt


def test_escaped_cue_text_reads_back_unchanged():
    raw = "R&D at AT&T <b>grew</b>"
    segments = [{"start": 0.0, "end": 2.0, "text": raw}]
    assert vtt_to_segments(segments_to_vtt(segments))[0]["text"] == raw


def test_srt_keeps_the_text_as_it_is():
    """SRT has no entity syntax, an escaped ampersand would show up as text."""
    srt = segments_to_srt([{"start": 0.0, "end": 2.0, "text": "R&D at AT&T"}])
    assert "R&D at AT&T" in srt


def test_a_blank_line_inside_a_segment_does_not_split_the_cue():
    """A cue ends at the first blank line, so the text must not contain one."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "line one\n\nline two"},
        {"start": 1.0, "end": 2.0, "text": "next"},
    ]
    for rendered in (segments_to_srt(segments), segments_to_vtt(segments)):
        assert "line one\nline two" in rendered
        assert len(vtt_to_segments(rendered)) == 2
