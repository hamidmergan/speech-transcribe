"""Subtitles: write segments to VTT and SRT, split long segments and read VTT back."""

import re
from typing import Dict, List

# Broadcast standard limits for one cue.
MAX_CHARS_PER_LINE = 42
MAX_LINES_PER_CUE = 2

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_time(seconds: float):
    total_ms = max(0, int(round(seconds * 1000)))
    h, total_ms = divmod(total_ms, 3_600_000)
    m, total_ms = divmod(total_ms, 60_000)
    s, ms = divmod(total_ms, 1000)
    return h, m, s, ms


def _fmt_vtt_time(seconds: float) -> str:
    """Format seconds as WebVTT timestamp HH:MM:SS.mmm."""
    h, m, s, ms = _split_time(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h, m, s, ms = _split_time(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_vtt(segments: List[Dict]) -> str:
    """Convert segments to a WebVTT string."""
    lines: List[str] = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_vtt_time(seg['start'])} --> {_fmt_vtt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def segments_to_srt(segments: List[Dict]) -> str:
    """Convert segments to an SRT string."""
    lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _parse_vtt_time(s: str) -> float:
    """Parse a WebVTT or SRT timestamp (HH:MM:SS.mmm or MM:SS.mmm) into seconds."""
    parts = s.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    return float(s)


def vtt_to_segments(raw: str) -> List[Dict]:
    """Parse a WebVTT (or SRT) string back into {start, end, text} segments."""
    segments: List[Dict] = []
    lines = raw.replace("\r", "").split("\n")
    i = 0
    while i < len(lines):
        arrow = lines[i].find("-->")
        if arrow == -1:
            i += 1
            continue
        try:
            start = _parse_vtt_time(lines[i][:arrow].strip())
            end = _parse_vtt_time(lines[i][arrow + 3:].strip().split(" ")[0])
        except (ValueError, IndexError):
            i += 1
            continue
        i += 1
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1
        if text_lines:
            segments.append({"start": start, "end": end, "text": " ".join(text_lines).strip()})
        i += 1
    return segments


def _lines(text: str, max_line: int) -> List[str]:
    """Pack words into lines, a single overlong word stays whole."""
    lines: List[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > max_line:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _pieces(text: str, max_line: int, max_lines: int) -> List[str]:
    """Split text into cues, preferably at sentence ends."""
    parts: List[List[str]] = []
    for sentence in _SENTENCE_END.split(text):
        if not sentence.strip():
            continue
        lines = _lines(sentence, max_line)
        for i in range(0, len(lines), max_lines):
            parts.append(lines[i:i + max_lines])
    merged: List[List[str]] = []
    for part in parts:
        # Two short sentences may share one cue.
        if merged and len(merged[-1]) + len(part) <= max_lines:
            merged[-1] = merged[-1] + part
        else:
            merged.append(part)
    return ["\n".join(p) for p in merged]


def split_long_segments(
    segments: List[Dict],
    max_line: int = MAX_CHARS_PER_LINE,
    max_lines: int = MAX_LINES_PER_CUE,
) -> List[Dict]:
    """Split long segments into short cues, the time shared by character count."""
    out: List[Dict] = []
    for seg in segments:
        text = " ".join(str(seg.get("text", "")).split())
        if not text:
            continue
        lines = _lines(text, max_line)
        if len(lines) <= max_lines:
            out.append({**seg, "text": "\n".join(lines)})
            continue
        pieces = _pieces(text, max_line, max_lines)
        total = sum(len(p) for p in pieces)
        start, end = float(seg["start"]), float(seg["end"])
        span = end - start
        done = 0
        boundary = start
        for i, piece in enumerate(pieces):
            done += len(piece)
            following = end if i == len(pieces) - 1 else start + span * done / total
            out.append({**seg, "start": boundary, "end": following, "text": piece})
            boundary = following
    return out
