"""Split long recogniser segments into short cues."""

from speech_transcribe.subtitles import split_long_segments

# A real-world style segment, 553 characters over 29.3 seconds.
_LONG = (
    "Responsibilities. At scientific conferences there are several people "
    "in charge. Depending on the size of the conference there are several leads, "
    "so-called chairs. They are experts from the respective field, usually "
    "professors, lecturers, but practitioners from companies can also be "
    "among them. Then there is always an organising committee. These are mostly "
    "staff members of the hosting institution. They are "
    "responsible for organisation and schedule, that means rooms, "
    "equipment, social programme, catering and so on and so forth."
)


def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


def test_short_segment_stays_unchanged():
    """What is already short the recogniser passes through."""
    segments = split_long_segments([_seg(1.0, 3.0, "Good morning everyone.")])
    assert segments == [_seg(1.0, 3.0, "Good morning everyone.")]


def test_long_segment_is_split():
    assert len(split_long_segments([_seg(0.0, 29.3, _LONG)])) > 1


def test_no_line_longer_than_the_limit():
    for seg in split_long_segments([_seg(0.0, 29.3, _LONG)]):
        for line in seg["text"].split("\n"):
            assert len(line) <= 42, line


def test_at_most_two_lines_per_cue():
    for seg in split_long_segments([_seg(0.0, 29.3, _LONG)]):
        assert len(seg["text"].split("\n")) <= 2


def test_no_word_is_cut():
    """All words come back out in the same order."""
    segments = split_long_segments([_seg(0.0, 29.3, _LONG)])
    out = " ".join(s["text"].replace("\n", " ") for s in segments).split()
    assert out == _LONG.split()


def test_times_stay_in_range_without_gap():
    segments = split_long_segments([_seg(10.0, 39.3, _LONG)])
    assert segments[0]["start"] == 10.0
    assert segments[-1]["end"] == 39.3
    for before, after in zip(segments, segments[1:]):
        assert before["end"] == after["start"]
        assert before["start"] < before["end"]


def test_split_at_sentence_end():
    """Two short sentences that together are too long split at the period."""
    text = (
        "The conference starts on Monday at nine in the large lecture hall. "
        "After that the talks of the working groups continue."
    )
    segments = split_long_segments([_seg(0.0, 8.0, text)])
    assert segments[0]["text"].replace("\n", " ").endswith("hall.")


def test_overlong_word_stays_whole():
    """No endless looping when a single word blows the limit."""
    word = "Danube" + "steamshippingcompany" * 5
    segments = split_long_segments([_seg(0.0, 4.0, word)])
    assert " ".join(s["text"].replace("\n", " ") for s in segments).split() == [word]


def test_empty_list_stays_empty():
    assert split_long_segments([]) == []
