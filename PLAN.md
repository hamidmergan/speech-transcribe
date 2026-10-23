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
- Every chunk is requested as `verbose_json`. Segments are shifted by the chunk start and merged. Text, VTT, SRT and JSON are all rendered from these segments, so every output format has the same timestamps.
- Defaults: chunk length 540 s (9 min), split at the nearest pause found by ffmpeg `silencedetect` (-40 dB, 0.3 s) within 60 s of the target. Chunks are re-encoded as 64 kbps mono MP3 at 16 kHz. Two chunks in parallel. Maximum file size 500 MB, maximum length 100 min. Read timeout 300 s.
- Transient server errors (5xx, 429, timeouts) are retried after 5, 15, 60, 180, 300, 420 and 600 s.
- The split search window is at most half a chunk, so a short custom chunk length can never move a split point backwards.
- Resume: the pipeline writes chunks only when the caller passes a cache key. `transcribe()` always passes one, built from the file content, the server address, the model, the language and the chunk length. Every finished chunk is stored as JSON in a work folder (default `~/.cache/speech-transcribe`). With `resume` a later run takes them from there. A fully successful run deletes them. An entry older than 7 days counts as absent, and every run with a key sweeps the folders that are past that age.
- Errors: every failure leaves the public `transcribe()` as a `TranscribeError`. ffmpeg, ffprobe, the file system and the HTTP library are wrapped at their own boundary, and an unexpected error is wrapped once more at the entry point. The CLI and the web worker both have a last-resort handler, so neither can end in a traceback or leave a job running.
- If a server answers a chunk with text but without segments, that chunk becomes one segment spanning the whole chunk.
- MP4 files that are fragmented or have their index at the end are remuxed losslessly before probing and splitting.
- Input files are checked by extension, size and leading bytes (`filetype`).
- Timestamps are rounded to whole milliseconds once, so a value like 1.9996 s becomes 00:00:02.000 and never 00:00:01.1000.
- API: `transcribe(path, language=None, fmt="txt", base_url=None, model=None, api_key=None, *, resume=False, on_progress=None, ...)` returns a `Transcript` with `.text`, `.segments`, `.language`, `.duration`, `.to_text()`, `.to_vtt()`, `.to_srt()`, `.to_json()` and `.render()`. `fmt` only picks what `render()` and `str()` return. `text` is accepted as an alias of `txt`.
- CLI follows doc-extract: one or several files, default output `<name>.<format>` next to the input, `-o` file or folder, `--stdout`, numbered names instead of overwriting. `--short-cues` splits long segments into cues of at most 2 lines of 42 characters. A failed chunk prints a hint to run again with `--resume`.
- Not carried: translation to English (the transcription page only offers transcription), the raw text/SRT/VTT server formats and their merge helpers (replaced by rendering from segments), the AI notice in VTT files, and the transcript-to-search-chunk offset index (it belongs to a search feature, not to transcription).

## Checklist

- [x] 0 Scaffold: repo, LICENSE (MIT), pyproject, package and tests folders, CI, PLAN.md, README skeleton, ffmpeg and ffprobe check
- [x] 1 Core: splitting, transcription per piece, offsets, merging, VTT/SRT output, MP4 repair, `config.py`, `errors.py`
- [x] 2 API and CLI: `transcribe(...) -> Transcript`, CLI with `--format`, `-o`, `--language`, `--resume`
- [x] 3 Tests with a fake server: splitting, offsets across pieces, resume, VTT and SRT, CLI, missing configuration
- [x] 4 README, CHANGELOG 0.1.0, local smoke test with a real speech-to-text model
- [x] 5 English sweep, acceptance greps
- [x] 6 Web UI (`speech-transcribe serve`) and Docker
- [x] 7 Review round: error contract, cache expiry, timeout construction, output and subtitle fixes

## Known properties

Behaviour taken from the source, kept on purpose, with no user visible harm.

- `--resume` still cuts and re-encodes every chunk before the cache is consulted. A long recording
  that needs one missing chunk re-encodes all of them, so resume saves the model time, not the
  ffmpeg time.
- The chunks are cut with input seek (`-ss` before `-i`). On a VBR MP3 or an MP4 without a precise
  index the cut lands on the frame boundary at or before the requested time, so a reported offset
  can run a few tens of milliseconds late at a chunk boundary.
- There is no cancel button on the page. A running transcription has to finish or fail.
- The page keeps one transcription at a time. A second upload while one runs is refused with 409.
- The chunks are butt joints at a speech pause, there is no overlap between them.

## Deliberate deviations from the source

- The split search window is at most half a chunk, so a short custom chunk length can never move a
  split point backwards. At the default 540 s the window is the unchanged 60 s.
- Timestamps are rounded to whole milliseconds once, so 1.9996 s becomes 00:00:02.000. The source
  rounded twice and could write 00:00:01.1000.
- SRT is rendered from the segments. The source only merged SRT that the server had already
  rendered.
- The failed-chunk message is "Chunk 3 transcription failed". The source wrote the kind of call
  into it ("Chunk 3 verbose transcription failed"), because it had a translation variant as well.
  This tool only transcribes, and "verbose" is a detail of the request, not of the recording.
- A chunk that answers with text but no segments becomes one segment spanning the chunk, so the
  run continues. The source raised on a missing segment list.
- WebVTT cue text is escaped (`&`, `<`, `>`). The source never escaped, because it only fed its own
  player. SRT has no entity syntax and stays unescaped.
- A segment is clamped to the end of its own chunk, so a model that writes past the cut cannot
  overlap the next chunk. The source did not clamp.
- The cache key covers the server address as well. The source keyed by job id, so the question
  never came up.
- The cache lives in a folder instead of Redis, and the 7 day Redis TTL became a 7 day age check
  with a sweep of stale folders.
- The MP4 helpers `is_fragmented_mp4`, `is_fragmented_mp4_reader` and `normalize_if_fragmented` are
  exported. The pipeline itself only uses `normalize_for_playback`.

## Open

- The page offers JSON next to text, SRT and VTT, because the library renders it from the same segments.
- There is no flag that deletes the whole work folder. The sweep and the 7 day age check do it over
  time, and the README says which folder to delete for an immediate clean-up.
