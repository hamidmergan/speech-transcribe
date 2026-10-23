# speech-transcribe plan

## What the tool does

Turns audio and video (MP3, WAV, M4A, MP4, WEBM, OGG, FLAC, MPEG) into text, VTT, SRT or JSON with timestamps.
Every piece of a recording goes to an OpenAI-compatible `/audio/transcriptions` endpoint (Whisper style, `verbose_json`).
Long recordings are split with ffmpeg at speech pauses, the timestamps of each piece are shifted by its start and merged.
An interrupted run resumes from the pieces that were already finished.

Shape: Python library with a thin CLI and a small web UI. pip package `speech-transcribe`, Python package `speech_transcribe`.
System requirements: ffmpeg and ffprobe.

## Decisions

- Provider: OpenAI-compatible interface only (`base_url`, `model`, optional `api_key`). No defaults for `base_url` and `model`.
- Python 3.10 or newer.

## Checklist

- [x] 0 Scaffold: repo, LICENSE (MIT), pyproject, package and tests folders, CI, PLAN.md, README skeleton, ffmpeg and ffprobe check
- [ ] 1 Core: splitting, transcription per piece, offsets, merging, VTT/SRT output, MP4 repair, `config.py`, `errors.py`
- [ ] 2 API and CLI: `transcribe(...) -> Transcript`, CLI with `--format`, `-o`, `--language`, `--resume`
- [ ] 3 Tests with a fake server: splitting, offsets across pieces, resume, VTT and SRT, CLI, missing configuration
- [ ] 4 README, CHANGELOG 0.1.0, local smoke test with a real speech-to-text model
- [ ] 5 English sweep, acceptance greps
- [ ] 6 Web UI (`speech-transcribe serve`) and Docker

## Open

- None yet.
