"""FFprobe-based media analysis for parallel video encoding."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path


_log = logging.getLogger("parallel-encoder")

FFPROBE_TIMEOUT_SECONDS = 30


def probe_file(path: str | Path) -> dict:
    """Run ffprobe on a single file and return a normalised info dict.

    Raises RuntimeError if ffprobe exits with a non-zero status, times out,
    or the output cannot be parsed.
    """
    path = Path(path).resolve()

    try:
        result: subprocess.CompletedProcess[bytes] = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(path),
            ],
            capture_output=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffprobe timed out after {FFPROBE_TIMEOUT_SECONDS}s for {path}"
        )

    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffprobe failed for {path} (exit {result.returncode}): {stderr_text}"
        )

    stdout_text = result.stdout.decode("utf-8", errors="replace")
    try:
        data: dict = json.loads(stdout_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f"Failed to parse ffprobe JSON for {path}") from exc

    fmt: dict = data.get("format", {})
    streams: list[dict] = data.get("streams", [])

    # --- video stream (first one found) ---
    video: dict | None = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )

    video_codec: str | None = None
    video_width: int | None = None
    video_height: int | None = None
    video_bitrate: int | None = None
    video_colour_primaries: str | None = None

    if video is not None:
        video_codec = video.get("codec_name")
        video_width = int(video["width"]) if "width" in video else None
        video_height = int(video["height"]) if "height" in video else None
        raw_br: str | None = video.get("bit_rate")
        video_bitrate = int(raw_br) if raw_br is not None else None
        video_colour_primaries = video.get("color_primaries")

    # --- audio streams ---
    audio_streams: list[dict] = []
    for s in streams:
        if s.get("codec_type") != "audio":
            continue
        tags: dict = s.get("tags", {})
        audio_streams.append(
            {
                "codec": s.get("codec_name", "und"),
                "language": tags.get("language", "und"),
                "channels": str(s.get("channels", "und")),
            }
        )

    # --- format-level fields ---
    total_bitrate_raw: str | None = fmt.get("bit_rate")
    total_bitrate: int | None = (
        int(total_bitrate_raw) if total_bitrate_raw is not None else None
    )

    duration_raw: str | None = fmt.get("duration")
    duration: float = float(duration_raw) if duration_raw is not None else 0.0

    file_size_raw: str | None = fmt.get("size")
    file_size: int = int(file_size_raw) if file_size_raw is not None else 0

    _log.debug(
        "Probed %s: codec=%s, %sx%s, bitrate=%s, duration=%.1fs",
        path.name, video_codec, video_width, video_height, total_bitrate, duration,
    )

    return {
        "path": str(path),
        "filename": path.stem,
        "file_size": file_size,
        "duration": duration,
        "video_codec": video_codec,
        "video_width": video_width,
        "video_height": video_height,
        "video_bitrate": video_bitrate,
        "video_colour_primaries": video_colour_primaries,
        "total_bitrate": total_bitrate,
        "audio_streams": audio_streams,
    }


_DEFAULT_EXTENSIONS: tuple[str, ...] = (
    "mp4", "m4v", "mkv", "avi", "mov", "wmv", "ts", "flv", "webm", "mpeg", "mpg",
)


def probe_folder(
    folder: str | Path,
    extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS,
) -> list[dict]:
    """Recursively scan *folder* for video files and probe each one.

    Returns a list of info dicts sorted by filename (stem, case-insensitive).
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise RuntimeError(f"Not a directory: {folder}")

    ext_lower: set[str] = {e.lower().lstrip(".") for e in extensions}
    files: list[Path] = sorted(
        (
            p
            for p in folder.rglob("*")
            if p.is_file()
            and not p.is_symlink()
            and p.suffix.lower().lstrip(".") in ext_lower
            and ".tmp." not in p.name
        ),
        key=lambda p: p.stem.lower(),
    )

    _log.info("Found %d video file(s) in %s", len(files), folder)

    results: list[dict] = []
    for f in files:
        results.append(probe_file(f))
    return results


def format_bitrate(bps: int | None) -> str:
    """Return a human-readable bitrate string."""
    if bps is None:
        return "N/A"
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.2f} Mb/s"
    if bps >= 1_000:
        return f"{bps / 1_000:.2f} Kb/s"
    return f"{bps} b/s"


def format_size(size_bytes: int) -> str:
    """Return a human-readable file-size string."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.2f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.2f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.2f} KB"
    return f"{size_bytes} B"
