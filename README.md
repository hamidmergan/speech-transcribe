# speech-transcribe

A lightweight Python utility for turning audio and video into text, WebVTT, SRT or JSON with timestamps. Each recording is sent to a speech-to-text model of your choice. Long recordings are split at speech pauses, transcribed piece by piece and merged back into one transcript with correct timestamps.

speech-transcribe currently supports:

- Audio (MP3, WAV, M4A, FLAC, OGG, MPGA)
- Video (MP4, WEBM, MPEG), the audio track is used

## Prerequisites

- Python 3.10 or newer
- `ffmpeg` and `ffprobe` on the PATH. They come together in the ffmpeg package (`apt install ffmpeg`, `brew install ffmpeg`).
- A server with an OpenAI-compatible API and a **speech-to-text model** (Whisper style, for example Ollama, vLLM, LM Studio, a university server or a cloud provider). A chat model is not enough. The server must answer `/audio/transcriptions` and understand `response_format=verbose_json`, otherwise there are no timestamps.
- An API key is optional. Many self-hosted servers do not need one.

If you run your own model, your recordings stay on your own network. Nothing leaves your machine or data center.

## Installation

```bash
pip install speech-transcribe
```

Alternatively, you can install it from the source:

```bash
git clone https://github.com/hamidmergan/speech-transcribe.git
cd speech-transcribe
pip install .
```

## Usage

### Command-Line

Print the result in the terminal:

```bash
speech-transcribe talk.mp4 --stdout --base-url http://localhost:11434/v1 --model your-speech-model
```

Write the result to a file instead:

```bash
speech-transcribe talk.mp4 --base-url http://localhost:11434/v1 --model your-speech-model
```

This writes `talk.txt` next to `talk.mp4`. Pick another format with `--format`:

```bash
speech-transcribe talk.mp4 --format vtt --base-url http://localhost:11434/v1 --model your-speech-model
speech-transcribe talk.mp4 --format srt -o subtitles/talk.srt --base-url ... --model ...
speech-transcribe talk.mp4 --format json --base-url ... --model ...
```

`txt` is the plain transcript, `vtt` and `srt` are subtitles, `json` holds the text, the language, the duration and every segment with its start and end.

Several files work the same way. Use `-o` to choose the output. For one file it is a file, for several files it is a folder:

```bash
speech-transcribe a.mp3 b.mp4 -o transcripts/ --base-url ... --model ...
```

If two inputs share the same name without extension (for example `a.mp3` and `a.mp4`), the full name is used instead (`a.mp3.txt`, `a.mp4.txt`). If the target already exists, a number is added (`talk (1).txt`, `talk (2).txt`). A file named explicitly with `-o` is overwritten. Errors appear on stderr. The exit code is 1 if at least one file failed.

Useful flags:

```bash
--language de     # the language of the recording, otherwise the server detects it
--resume          # continue an interrupted run from the pieces that are already done
--short-cues      # split long segments into cues of at most 2 lines of 42 characters
--quiet           # no progress on stderr
```

### Resuming

A long recording is split into pieces and every finished piece is kept in a work folder. If a piece fails, the run stops and says so. Start it again with `--resume` and only the missing pieces are sent:

```bash
speech-transcribe lecture.mp4 --resume --base-url ... --model ...
```

A run that finishes completely empties the work folder again.

### Python API

```python
from speech_transcribe import transcribe

transcript = transcribe(
    "talk.mp4",
    base_url="http://localhost:11434/v1",
    model="your-speech-model",
)
print(transcript.text)
```

The result carries everything, so you can render any format without transcribing again:

```python
transcript.text            # the plain transcript
transcript.segments        # [{"start": 0.0, "end": 2.5, "text": "..."}, ...]
transcript.language        # the language the server detected
transcript.duration        # the length of the recording in seconds
transcript.to_vtt()        # WebVTT
transcript.to_srt()        # SRT
transcript.to_json()       # text, language, duration and segments
```

`to_vtt(short_cues=True)` and `to_srt(short_cues=True)` split long segments into readable cues.

Long recordings with progress and resume:

```python
transcript = transcribe(
    "lecture.mp4",
    language="de",
    base_url="http://localhost:11434/v1",
    model="your-speech-model",
    resume=True,
    on_progress=lambda done, total: print(f"{done}/{total} pieces done"),
)
```

### Settings

Order: argument, then environment variable, then default.

| Argument / flag | Environment variable | Default |
| --- | --- | --- |
| `base_url` / `--base-url` | `SPEECH_TRANSCRIBE_BASE_URL` | none, required |
| `model` / `--model` | `SPEECH_TRANSCRIBE_MODEL` | none, required |
| `api_key` / `--api-key` | `SPEECH_TRANSCRIBE_API_KEY` (then `OPENAI_API_KEY`) | optional |
| `language` / `--language` | `SPEECH_TRANSCRIBE_LANGUAGE` | detected by the server |
| `chunk_seconds` / `--chunk-seconds` | `SPEECH_TRANSCRIBE_CHUNK_SECONDS` | 540 seconds |
| `max_file_mb` / `--max-file-mb` | `SPEECH_TRANSCRIBE_MAX_FILE_MB` | 500 |
| `max_duration_min` / `--max-duration-min` | `SPEECH_TRANSCRIBE_MAX_DURATION_MIN` | 100 minutes |
| `concurrency` / `--concurrency` | `SPEECH_TRANSCRIBE_CONCURRENCY` | 2 |
| `timeout` / `--timeout` | `SPEECH_TRANSCRIBE_TIMEOUT` | 300 seconds |
| `work_dir` / `--work-dir` | `SPEECH_TRANSCRIBE_WORK_DIR` | `~/.cache/speech-transcribe` |

If a limit is exceeded, the file is rejected with a message naming the limit and the setting. Nothing is silently truncated. A recording longer than `chunk_seconds` is split, every piece is re-encoded as a small mono MP3 before it is sent, so the upload stays small.

## Limitations

Quality depends on the chosen model. A small model may misread names, technical terms, dialect and recordings with background noise. Check important results.

Timestamps are only as good as the server's answer. A server that cannot deliver `verbose_json` returns one segment covering the whole piece, so `vtt` and `srt` stay coarse.

Speakers are not separated. There is no diarization and no speaker labels.

## Running Tests

```bash
pip install -e ".[test]"
pytest
```

The tests run against a local fake server and need no real model. The tests that cut real media need ffmpeg and are skipped without it.

## License

MIT
