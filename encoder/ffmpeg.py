"""Build and run FFmpeg commands for parallel video encoding."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

_log = logging.getLogger("parallel-encoder")


@dataclass
class EncodingResult:
    """Result of an FFmpeg encoding run."""

    source_path: str
    output_path: str
    success: bool
    exit_code: int
    encoding_time: float
    error_message: str | None


def find_ffmpeg() -> str:
    """Locate the ffmpeg binary on PATH.

    Returns:
        Absolute path to the ffmpeg binary.

    Raises:
        RuntimeError: If ffmpeg is not found on PATH.
    """
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Please install ffmpeg or add it to your PATH."
        )
    return path


def find_ffprobe() -> str:
    """Locate the ffprobe binary on PATH.

    Returns:
        Absolute path to the ffprobe binary.

    Raises:
        RuntimeError: If ffprobe is not found on PATH.
    """
    path = shutil.which("ffprobe")
    if path is None:
        raise RuntimeError(
            "ffprobe not found on PATH. Please install ffprobe or add it to your PATH."
        )
    return path


def atomic_output_path(output: str) -> str:
    """Return a temporary path next to the final output.

    The temp file sits in the same directory so os.replace is atomic
    (same filesystem).
    """
    p = Path(output)
    return str(p.with_suffix(f".tmp{p.suffix}"))


def finalize_output(temp_path: str, final_path: str) -> None:
    """Atomically move the temp file to its final location."""
    os.replace(temp_path, final_path)


def cleanup_temp(temp_path: str) -> None:
    """Remove a temp file if it exists (best-effort)."""
    try:
        os.unlink(temp_path)
    except OSError as exc:
        _log.debug("Could not remove temp file %s: %s", temp_path, exc)


def build_command(
    ffmpeg_path: str,
    source: str,
    output: str,
    preset_args: list[str],
    threads: int,
    test_encode: dict | None = None,
) -> list[str]:
    """Construct the full ffmpeg command as a list of strings.

    Args:
        ffmpeg_path: Path to the ffmpeg binary.
        source: Path to the source video file.
        output: Path for the output file.
        preset_args: Codec and encoding arguments (e.g. ["-c:v", "libx265", "-crf", "22"]).
        threads: Number of threads to use for encoding.
        test_encode: Optional dict with ``start_seconds`` and ``duration_seconds``
            keys for a partial test encode.

    Returns:
        Complete ffmpeg command as a list of strings.
    """
    command: list[str] = [ffmpeg_path, "-y", "-hide_banner"]

    # Input arguments -------------------------------------------------------
    input_args: list[str] = []
    if test_encode is not None:
        input_args.extend(["-ss", str(test_encode["start_seconds"])])
        input_args.extend(["-t", str(test_encode["duration_seconds"])])

    command.extend(input_args)
    command.extend(["-i", source])

    # For test encodes, strip source metadata so the output container
    # reports the actual encoded duration, not the original file's duration.
    if test_encode is not None:
        command.extend(["-map_metadata", "-1"])

    # Preset arguments (copy so we don't mutate the caller's list) ----------
    args = list(preset_args)

    # Thread control --------------------------------------------------------
    effective_threads = threads
    if "libx265" in args:
        effective_threads = min(threads, _X265_MAX_THREADS)
    thread_args: list[str] = ["-threads", str(effective_threads)]

    if "libx265" in args:
        pools_param = _x265_pools_param(threads)
        try:
            idx = args.index("-x265-params")
            args[idx + 1] = args[idx + 1] + ":" + pools_param
        except (ValueError, IndexError):
            args.extend(["-x265-params", pools_param])

    elif "libsvtav1" in args:
        lp_param = f"lp={threads}"
        try:
            idx = args.index("-svtav1-params")
            args[idx + 1] = args[idx + 1] + ":" + lp_param
        except (ValueError, IndexError):
            args.extend(["-svtav1-params", lp_param])

    elif "libx264" in args:
        args.extend(["-x264-params", f"threads={threads}"])

    elif "libvpx-vp9" in args:
        args.extend(["-tile-columns", "2", "-tile-rows", "1"])
        # -threads is already added via thread_args

    command.extend(args)
    command.extend(thread_args)
    command.append(output)

    return command


_X265_MAX_THREADS = 16


def _x265_pools_param(threads: int) -> str:
    """Build the x265 ``pools`` parameter.

    x265 has a hard limit on frame-threads (typically 16). Requesting
    more causes it to refuse to start. We cap the pool size accordingly.
    """
    capped = min(threads, _X265_MAX_THREADS)
    return f"pools={capped}"


# ---------------------------------------------------------------------------
# Progress parsing
# ---------------------------------------------------------------------------

_PROGRESS_RE = re.compile(
    r"frame=\s*(?P<frame>\d+)\s+"
    r"fps=\s*(?P<fps>[\d.]+)\s+"
    r".*?"
    r"size=\s*(?P<size>\S+)\s+"
    r"time=\s*(?P<time>\S+)\s+"
    r"bitrate=\s*(?P<bitrate>\S+)\s+"
    r".*?"
    r"speed=\s*(?P<speed>[\d.]+)x"
)


def _parse_time(time_str: str) -> float:
    """Convert an FFmpeg time string ``HH:MM:SS.ss`` to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return 0.0


def _parse_progress_line(line: str) -> dict | None:
    """Parse an FFmpeg progress line into a dict, or return *None*."""
    match = _PROGRESS_RE.search(line)
    if match is None:
        return None
    return {
        "frame": int(match.group("frame")),
        "fps": float(match.group("fps")),
        "time_seconds": _parse_time(match.group("time")),
        "speed": float(match.group("speed")),
        "bitrate": match.group("bitrate"),
        "size": match.group("size"),
    }


# ---------------------------------------------------------------------------
# Encoding runner
# ---------------------------------------------------------------------------


def run_encode(
    command: list[str],
    progress_callback: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> EncodingResult:
    """Launch an FFmpeg encode subprocess and monitor its output.

    Args:
        command: The full ffmpeg command (as returned by :func:`build_command`).
        progress_callback: Optional callable invoked with a progress dict on
            every FFmpeg progress line.

    Returns:
        An :class:`EncodingResult` describing the outcome.
    """
    source = ""
    output = ""
    # Extract source and output from the command list
    try:
        i_idx = command.index("-i")
        source = command[i_idx + 1]
    except (ValueError, IndexError):
        pass
    if command:
        output = command[-1]

    # Write to temp file, rename on success
    temp = atomic_output_path(output)
    command = list(command)  # copy to avoid mutating caller
    command[-1] = temp

    start_time = time.monotonic()
    stderr_lines: list[str] = []
    process: subprocess.Popen[str] | None = None

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stderr is not None  # for type checkers

        _log.debug("FFmpeg command: %s", " ".join(command))

        for line in process.stderr:
            if cancel_event is not None and cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                cleanup_temp(temp)
                encoding_time = time.monotonic() - start_time
                return EncodingResult(
                    source_path=source,
                    output_path=output,
                    success=False,
                    exit_code=-1,
                    encoding_time=encoding_time,
                    error_message="Encoding cancelled.",
                )

            line = line.rstrip("\n\r")
            stderr_lines.append(line)

            if progress_callback is not None:
                progress = _parse_progress_line(line)
                if progress is not None:
                    progress_callback(progress)

        process.wait()
        encoding_time = time.monotonic() - start_time

        success = process.returncode == 0
        error_message: str | None = None

        if success:
            finalize_output(temp, output)
            _log.debug("FFmpeg completed: exit=%d time=%.1fs source=%s", process.returncode, encoding_time, source)
            _log.debug("FFmpeg stderr for %s:\n%s", source, "\n".join(stderr_lines[-50:]))
        else:
            cleanup_temp(temp)
            full_stderr = "\n".join(stderr_lines[-20:])
            # Extract just the last meaningful line for the short message
            last_line = ""
            for ln in reversed(stderr_lines):
                ln = ln.strip()
                if ln and not ln.startswith("frame="):
                    last_line = ln
                    break
            error_message = last_line or f"exit code {process.returncode}"
            _log.warning("FFmpeg failed for %s (exit %d): %s", source, process.returncode, error_message)
            _log.debug("FFmpeg full stderr for %s:\n%s", source, full_stderr)

        return EncodingResult(
            source_path=source,
            output_path=output,
            success=success,
            exit_code=process.returncode,
            encoding_time=encoding_time,
            error_message=error_message,
        )

    except KeyboardInterrupt:
        encoding_time = time.monotonic() - start_time
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        cleanup_temp(temp)
        return EncodingResult(
            source_path=source,
            output_path=output,
            success=False,
            exit_code=-1,
            encoding_time=encoding_time,
            error_message="Encoding interrupted by user.",
        )
