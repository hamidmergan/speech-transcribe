# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Finished pieces expire after 7 days, and every run deletes the folders that are past that age
- The number of pieces a server answered with nothing is reported on stderr and on the page
- `to_json(short_cues=True)` and `--short-cues --format json` now split long segments as well
- The MP4 helpers `is_fragmented_mp4`, `is_fragmented_mp4_reader`, `normalize_if_fragmented` and `normalize_for_playback` are exported
- `.dockerignore`, so the build context carries neither the virtualenv nor the git history

### Changed

- Every error leaves the public API as a `TranscribeError`, so neither the command line nor the web page can end in a raw traceback
- The cache is written only when the caller passes a cache key, and the key now covers the server address as well
- `-o` with a single input writes into the folder when it names one, instead of failing after the transcription
- WebVTT cue text is escaped, and a blank line inside a segment no longer splits a cue
- A segment that reaches past its piece is clamped to the cut, so pieces never overlap
- The too-long message names the recording length with one decimal

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
- Web page (`speech-transcribe serve`): drop a recording, pick the format, follow the parts, edit and download the transcript
- Dockerfile with ffmpeg and a docker-compose.yml, `docker compose up` serves the page on port 8000
- Works with any OpenAI-compatible server, with or without an API key

[Unreleased]: https://github.com/hamidmergan/speech-transcribe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hamidmergan/speech-transcribe/releases/tag/v0.1.0
