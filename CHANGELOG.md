# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-23

### Added

- Transcription of audio and video (MP3, WAV, M4A, FLAC, OGG, MPGA, MP4, WEBM, MPEG) with any OpenAI-compatible speech-to-text model
- Output as plain text, WebVTT, SRT or JSON, all rendered from the same segments
- Long recordings are split with ffmpeg at speech pauses, the timestamps of every piece are shifted and merged
- Resume: finished pieces are kept in a work folder, `--resume` continues an interrupted run
- Fragmented MP4 files and files with the index at the end are remuxed losslessly before splitting
- Retries for temporary server errors (5xx, 429, timeouts)
- `--short-cues` splits long segments into cues of at most 2 lines of 42 characters
- Limits for file size and recording length with clear error messages
- Command-line tool with output to file, folder or stdout, and progress per piece
- Python API (`transcribe`, `Transcript`)
- Works with any OpenAI-compatible server, with or without an API key

[Unreleased]: https://github.com/hamidmergan/speech-transcribe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hamidmergan/speech-transcribe/releases/tag/v0.1.0
