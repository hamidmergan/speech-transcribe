import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .core import FORMATS, normalize_format, output_name, transcribe
from .errors import ConfigError, MissingToolError, TranscribeError

LOOPBACK = ("127.0.0.1", "localhost", "::1")
UI_HINT = 'The web page needs extra packages: pip install "speech-transcribe[ui]"'
NO_LOGIN = "Warning: the page has no login. Anyone who can reach this port can use it."


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", help="URL of the OpenAI-compatible server, e.g. http://localhost:11434/v1")
    p.add_argument("--model", help="Name of the speech-to-text model")
    p.add_argument("--api-key", help="API key, if the server requires one")
    p.add_argument("--language", help="Language of the recording as ISO 639-1 code, e.g. en. Default: detected")
    p.add_argument("--chunk-seconds", type=float, help="Length of one chunk in seconds (default: 540)")
    p.add_argument("--max-file-mb", type=float, help="Largest accepted file in MB (default: 500)")
    p.add_argument("--max-duration-min", type=float, help="Longest accepted recording in minutes (default: 100)")
    p.add_argument("--concurrency", type=int, help="Chunks sent to the server at once (default: 2)")
    p.add_argument("--timeout", type=float, help="Read timeout per request in seconds (default: 300)")
    p.add_argument("--work-dir", help="Folder for finished chunks (default: ~/.cache/speech-transcribe)")


_OPTION_KEYS = (
    "base_url", "model", "api_key", "language", "chunk_seconds",
    "max_file_mb", "max_duration_min", "concurrency", "timeout", "work_dir",
)


def _serve_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="speech-transcribe serve", description="Start the local web page.")
    p.add_argument("--host", default="127.0.0.1", help="Default: 127.0.0.1 (this computer only)")
    p.add_argument("--port", type=int, default=8000)
    _model_args(p)
    return p


def _serve(argv: List[str]) -> int:
    args = _serve_parser().parse_args(argv)
    try:
        import uvicorn

        from .web.app import create_app
    except ImportError:
        print(UI_HINT, file=sys.stderr)
        return 2
    options = {key: getattr(args, key) for key in _OPTION_KEYS}
    try:
        app = create_app(
            # on loopback, only accept our own host names, so other web pages cannot reach it via DNS rebinding
            allowed_hosts=list(LOOPBACK) if args.host in LOOPBACK else None,
            **options,
        )
    except TranscribeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if args.host not in LOOPBACK:
        print(NO_LOGIN, file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="speech-transcribe",
        description="Transcribe audio and video into text, VTT, SRT or JSON with timestamps "
        "through an OpenAI-compatible speech-to-text server.",
        epilog="Run 'speech-transcribe serve' for the web page.",
    )
    p.add_argument("files", nargs="+", help="Audio or video files (MP3, WAV, M4A, MP4, WEBM, OGG, FLAC, MPEG)")
    p.add_argument("-f", "--format", default="txt", choices=FORMATS, help="Output format (default: txt)")
    p.add_argument(
        "-o",
        "--output",
        help="Output file (one input) or output folder (several inputs). "
        "Default: write <name>.<format> next to each input",
    )
    p.add_argument("--stdout", action="store_true", help="Print the result instead of writing files")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from the chunks that were already finished",
    )
    p.add_argument(
        "--short-cues",
        action="store_true",
        help="Split long segments into cues of at most 2 lines of 42 characters (vtt, srt)",
    )
    _model_args(p)
    p.add_argument("-q", "--quiet", action="store_true", help="Do not print progress")
    p.add_argument("--version", action="version", version=f"speech-transcribe {__version__}")
    return p


def _free(path: Path) -> Path:
    """Return path, or 'name (1).txt', 'name (2).txt', ... if it already exists."""
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        n += 1
    return candidate


def _targets(keys: List[str], fmt: str, out: Optional[Path]) -> dict:
    """Map each input to its output file: <stem>.<ext>, or <name>.<ext> if two stems would collide."""
    stems = [Path(k).stem for k in keys]
    unique = len(set(stems)) == len(stems)
    targets = {}
    for key in keys:
        path = Path(key)
        name = output_name(key, fmt) if unique else path.name + Path(output_name(key, fmt)).suffix
        targets[key] = (out / name) if out is not None else path.with_name(name)
    return targets


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        return _serve(argv[1:])
    args = _parser().parse_args(argv)
    fmt = normalize_format(args.format)
    many = len(args.files) > 1
    out = Path(args.output) if args.output else None

    if args.stdout and out is not None:
        print("--stdout and -o cannot be combined.", file=sys.stderr)
        return 2
    if many and out is not None and out.exists() and not out.is_dir():
        print("With several files, -o must be a folder.", file=sys.stderr)
        return 2

    targets = _targets(args.files, fmt, out if many else None)
    if many and out is not None:
        out.mkdir(parents=True, exist_ok=True)

    failed = 0
    for key in args.files:
        name = Path(key).name

        def progress(done: int, total: int, name=name) -> None:
            if not args.quiet:
                print(f"{name}: chunk {done}/{total} done", file=sys.stderr)

        try:
            transcript = transcribe(
                key,
                language=args.language,
                fmt=fmt,
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
                resume=args.resume,
                on_progress=progress,
                chunk_seconds=args.chunk_seconds,
                max_file_mb=args.max_file_mb,
                max_duration_min=args.max_duration_min,
                concurrency=args.concurrency,
                timeout=args.timeout,
                work_dir=args.work_dir,
            )
        except (ConfigError, MissingToolError) as exc:
            # Nothing can work without these, so stop at once.
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        except TranscribeError as exc:
            failed += 1
            print(f"Error for {key}: {exc}", file=sys.stderr)
            if getattr(exc, "resumable", False):
                print("Finished chunks were kept. Run again with --resume to continue.", file=sys.stderr)
            continue
        except Exception as exc:  # no raw traceback ever reaches the terminal
            failed += 1
            print(f"Error for {key}: {exc}", file=sys.stderr)
            continue

        result = transcript.render(fmt, short_cues=args.short_cues)
        if args.stdout:
            if many:
                print(f"== {name} ==")
            print(result)
            if many:
                print()
            continue
        # an explicitly named output file is overwritten, generated names never are
        target = out if (out is not None and not many) else _free(targets[key])
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result if result.endswith("\n") else result + "\n", encoding="utf-8")
        except OSError as exc:
            # the transcription is already paid for, so say what went wrong
            failed += 1
            print(f"Could not write {target}: {exc}", file=sys.stderr)
            continue
        if not args.quiet:
            print(f"Wrote {target}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
