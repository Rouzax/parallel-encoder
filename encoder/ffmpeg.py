"""Build and run FFmpeg commands for parallel video encoding."""

from __future__ import annotations

import logging
import os
import platform
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


def extract_cover_art(
    ffmpeg_path: str,
    source: str,
    cover_art: list[dict],
    temp_dir: str,
) -> list[tuple[str, str, str]]:
    """Extract cover art streams from source to temp files.

    Args:
        ffmpeg_path: Path to ffmpeg binary.
        source: Source video file path.
        cover_art: List of dicts with ``index``, ``filename``, ``mimetype``.
        temp_dir: Directory to write extracted files.

    Returns:
        List of (temp_file_path, original_filename, mimetype) tuples.
    """
    extracted: list[tuple[str, str, str]] = []
    for i, art in enumerate(cover_art):
        stream_idx = art["index"]
        orig_name = art["filename"]
        mimetype = art["mimetype"]
        # Use unique temp name to avoid collisions (multiple covers may share names)
        ext = Path(orig_name).suffix or ".png"
        temp_path = str(Path(temp_dir) / f"_cover_{i}{ext}")
        try:
            result = subprocess.run(
                [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
                 "-i", source, "-map", f"0:{stream_idx}", "-frames:v", "1", temp_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and Path(temp_path).exists():
                extracted.append((temp_path, orig_name, mimetype))
                _log.debug("Extracted cover art: stream %d -> %s (%s)", stream_idx, orig_name, mimetype)
            else:
                _log.warning("Failed to extract cover art stream %d: %s", stream_idx, result.stderr.strip())
        except Exception as exc:
            _log.warning("Error extracting cover art stream %d: %s", stream_idx, exc)
    return extracted


def cover_art_attach_args(extracted: list[tuple[str, str, str]]) -> list[str]:
    """Build FFmpeg ``-attach`` arguments for extracted cover art files.

    Args:
        extracted: List of (temp_file_path, original_filename, mimetype) tuples
            as returned by :func:`extract_cover_art`.
    """
    args: list[str] = []
    for i, (temp_path, orig_name, mimetype) in enumerate(extracted):
        args.extend(["-attach", temp_path])
        args.extend([f"-metadata:s:t:{i}", f"mimetype={mimetype}"])
        args.extend([f"-metadata:s:t:{i}", f"filename={orig_name}"])
    return args


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

    uses_internal_threading = False

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
        uses_internal_threading = True

    elif "libvpx-vp9" in args:
        args.extend(["-tile-columns", "2", "-tile-rows", "1"])
        # -threads is already added via thread_args

    command.extend(args)
    if not uses_internal_threading:
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

_STALL_WARNING_SECONDS = 300  # 5 minutes without progress -> log warning

_PROGRESS_RE = re.compile(
    r"frame=\s*(?P<frame>\d+)\s+"
    r"fps=\s*(?P<fps>[\d.]+)\s+"
    r".*?"
    r"L?size=\s*(?P<size>\S+)\s+"
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
    _log.debug("Unexpected time format from FFmpeg: %s", time_str)
    return 0.0


def _parse_progress_line(line: str) -> dict | None:
    """Parse an FFmpeg progress line into a dict, or return *None*."""
    match = _PROGRESS_RE.search(line)
    if match is None:
        return None
    return {
        "frame": int(match.group("frame")),
        "fps": float(match.group("fps")),
        "time_seconds": max(0.0, _parse_time(match.group("time"))),
        "speed": float(match.group("speed")),
        "bitrate": match.group("bitrate"),
        "size": match.group("size"),
    }


# ---------------------------------------------------------------------------
# Windows NUMA affinity
# ---------------------------------------------------------------------------


def _try_set_cpu_set_masks(kernel32, handle, affinity, process_id: int, numa_node: int) -> None:  # type: ignore[no-untyped-def]
    """Best-effort SetProcessDefaultCpuSetMasks call (Server 2022+ / Win11+).

    Constrains all future threads of the process to the given NUMA node,
    preventing codecs with internal NUMA-aware thread creation (e.g.
    SVT-AV1) from spawning threads in other processor groups.
    """
    import ctypes
    import ctypes.wintypes

    try:
        GROUP_AFFINITY = type(affinity)
        kernel32.SetProcessDefaultCpuSetMasks.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(GROUP_AFFINITY),
            ctypes.c_ushort,
        ]
        kernel32.SetProcessDefaultCpuSetMasks.restype = ctypes.wintypes.BOOL
        if kernel32.SetProcessDefaultCpuSetMasks(
            handle, ctypes.byref(affinity), ctypes.c_ushort(1),
        ):
            _log.debug(
                "SetProcessDefaultCpuSetMasks OK for PID %d, NUMA node %d",
                process_id, numa_node,
            )
    except (AttributeError, OSError):
        pass  # Not available on this Windows version


def _set_windows_process_numa(process_handle: int, process_id: int, numa_node: int) -> bool:
    """Pin a Windows process to a NUMA node.

    Uses SetProcessAffinityMask (all Windows versions, affects all existing
    threads) as the primary method.  Falls back to per-thread group affinity
    for cross-group NUMA nodes, and to SetProcessDefaultCpuSetMasks
    (Server 2022+ / Win11+) for future threads.
    """
    import ctypes
    import ctypes.wintypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    class GROUP_AFFINITY(ctypes.Structure):
        _fields_ = [
            ("Mask", ctypes.c_ulonglong),
            ("Group", ctypes.wintypes.WORD),
            ("Reserved", ctypes.wintypes.WORD * 3),
        ]

    # -- Resolve NUMA node to processor group + mask ----------------------
    affinity = GROUP_AFFINITY()
    ctypes.memset(ctypes.byref(affinity), 0, ctypes.sizeof(GROUP_AFFINITY))

    kernel32.GetNumaNodeProcessorMaskEx.argtypes = [
        ctypes.c_ushort, ctypes.POINTER(GROUP_AFFINITY),
    ]
    kernel32.GetNumaNodeProcessorMaskEx.restype = ctypes.wintypes.BOOL

    if not kernel32.GetNumaNodeProcessorMaskEx(
        ctypes.c_ushort(numa_node), ctypes.byref(affinity),
    ):
        _log.warning("GetNumaNodeProcessorMaskEx failed for NUMA node %d", numa_node)
        return False

    target_group = affinity.Group
    target_mask = affinity.Mask
    _log.debug(
        "NUMA node %d -> group=%d, mask=0x%x (%d processors)",
        numa_node, target_group, target_mask, bin(target_mask).count("1"),
    )

    handle = ctypes.wintypes.HANDLE(process_handle)

    # -- Determine the process's current processor group ------------------
    # New processes inherit their parent's group (typically group 0).
    # SetProcessAffinityMask silently applies the mask within the process's
    # current group even when the target is a different group, so we must
    # detect cross-group situations and use per-thread pinning instead.
    process_group = _get_process_group(kernel32, handle)

    # -- SetProcessDefaultCpuSetMasks for future threads --------------------
    # Called for ALL NUMA nodes (same-group and cross-group) to prevent
    # codecs like SVT-AV1 from creating threads in other processor groups
    # via their own internal NUMA-aware thread management.
    # Server 2022+ / Windows 11+ only.
    _try_set_cpu_set_masks(kernel32, handle, affinity, process_id, numa_node)

    if target_group == process_group:
        # -- Same group: SetProcessAffinityMask (all existing threads) ----
        kernel32.SetProcessAffinityMask.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.c_size_t,
        ]
        kernel32.SetProcessAffinityMask.restype = ctypes.wintypes.BOOL

        if kernel32.SetProcessAffinityMask(handle, target_mask):
            _log.info(
                "Pinned process (PID %d) to NUMA node %d via "
                "SetProcessAffinityMask + SetProcessDefaultCpuSetMasks "
                "(group=%d, mask=0x%x)",
                process_id, numa_node, target_group, target_mask,
            )
            return True
        _log.warning(
            "SetProcessAffinityMask failed for PID %d, NUMA node %d",
            process_id, numa_node,
        )
        return False

    # -- Cross-group: per-thread SetThreadGroupAffinity -------------------
    # SetProcessAffinityMask cannot move threads to a different processor
    # group, so we enumerate threads and pin each one individually.
    pinned = _pin_threads_to_group(
        kernel32, process_id, target_group, target_mask, GROUP_AFFINITY,
    )

    # Second pass: catch any threads created between our first enumeration
    # and SetProcessDefaultCpuSetMasks taking effect.
    pinned += _pin_threads_to_group(
        kernel32, process_id, target_group, target_mask, GROUP_AFFINITY,
    )

    _log.info(
        "Pinned %d thread(s) of PID %d to NUMA node %d via "
        "SetThreadGroupAffinity (group=%d, mask=0x%x)",
        pinned, process_id, numa_node, target_group, target_mask,
    )

    return pinned > 0


def _get_process_group(kernel32, handle) -> int:  # type: ignore[no-untyped-def]
    """Return the processor group a process currently belongs to.

    Uses GetProcessGroupAffinity which returns the list of groups the
    process is associated with.  For a freshly spawned process this is
    typically a single group inherited from the parent.  Returns 0 on
    failure (safe default — new processes start in group 0).
    """
    import ctypes
    import ctypes.wintypes

    kernel32.GetProcessGroupAffinity.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.POINTER(ctypes.wintypes.USHORT),
        ctypes.POINTER(ctypes.wintypes.USHORT),
    ]
    kernel32.GetProcessGroupAffinity.restype = ctypes.wintypes.BOOL

    group_count = ctypes.wintypes.USHORT(1)
    groups = (ctypes.wintypes.USHORT * 1)()

    if kernel32.GetProcessGroupAffinity(handle, ctypes.byref(group_count), groups):
        return groups[0]
    return 0


def _pin_threads_to_group(
    kernel32,  # type: ignore[no-untyped-def]
    process_id: int,
    target_group: int,
    target_mask: int,
    group_affinity_cls: type,
) -> int:
    """Enumerate threads of *process_id* and pin each to *target_group*/*target_mask*.

    Returns the number of threads successfully pinned.

    Note: argtypes are intentionally NOT set on kernel32 functions here.
    Multiple worker threads call this concurrently, and setting argtypes
    on the shared kernel32 object with locally-defined ctypes Structure
    classes causes race conditions (LP_THREADENTRY32 type mismatch).
    """
    import ctypes
    import ctypes.wintypes

    TH32CS_SNAPTHREAD = 0x00000004
    THREAD_SET_INFORMATION = 0x0020
    THREAD_QUERY_INFORMATION = 0x0040
    INVALID_HANDLE = ctypes.wintypes.HANDLE(-1).value

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.wintypes.DWORD),
            ("cntUsage", ctypes.wintypes.DWORD),
            ("th32ThreadID", ctypes.wintypes.DWORD),
            ("th32OwnerProcessID", ctypes.wintypes.DWORD),
            ("tpBasePri", ctypes.wintypes.LONG),
            ("tpDeltaPri", ctypes.wintypes.LONG),
            ("dwFlags", ctypes.wintypes.DWORD),
        ]

    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if snap == INVALID_HANDLE:
        _log.warning("CreateToolhelp32Snapshot failed")
        return 0

    new_affinity = group_affinity_cls()
    ctypes.memset(ctypes.byref(new_affinity), 0, ctypes.sizeof(group_affinity_cls))
    new_affinity.Mask = target_mask
    new_affinity.Group = target_group

    te = THREADENTRY32()
    te.dwSize = ctypes.sizeof(THREADENTRY32)
    pinned = 0

    try:
        if not kernel32.Thread32First(snap, ctypes.byref(te)):
            return 0
        while True:
            if te.th32OwnerProcessID == process_id:
                th = kernel32.OpenThread(
                    THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION,
                    False, te.th32ThreadID,
                )
                if th:
                    if kernel32.SetThreadGroupAffinity(
                        th, ctypes.byref(new_affinity), None,
                    ):
                        pinned += 1
                    else:
                        _log.debug(
                            "SetThreadGroupAffinity failed for thread %d",
                            te.th32ThreadID,
                        )
                    kernel32.CloseHandle(th)
            if not kernel32.Thread32Next(snap, ctypes.byref(te)):
                break
    finally:
        kernel32.CloseHandle(snap)

    return pinned


# ---------------------------------------------------------------------------
# Encoding runner
# ---------------------------------------------------------------------------


def run_encode(
    command: list[str],
    progress_callback: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    numa_node: int | None = None,
    threads_per_numa: int | None = None,
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

        # Pin to NUMA node on Windows (Linux uses numactl prefix in _wrap_numa)
        if numa_node is not None and platform.system() == "Windows":
            try:
                _set_windows_process_numa(process._handle, process.pid, numa_node)  # type: ignore[union-attr]
            except (AttributeError, OSError) as exc:
                _log.warning("Windows NUMA pinning failed: %s", exc)

        _log.debug("FFmpeg command: %s", " ".join(command))

        # Use readline() instead of the file iterator — the iterator
        # buffers aggressively on Windows and never yields \r-terminated
        # progress lines from FFmpeg.
        last_progress_time = time.monotonic()
        stall_warned = False

        while True:
            line = process.stderr.readline()
            if not line:
                break

            if cancel_event is not None and cancel_event.is_set():
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
                    last_progress_time = time.monotonic()
                    stall_warned = False
                    progress_callback(progress)
                elif "frame=" in line and len(stderr_lines) <= 5:
                    # Log unparsed progress lines for diagnostics
                    _log.debug("Unparsed progress line: %s", line[:300])

            # Warn on stall (but don't kill — scene may be legitimately slow)
            elapsed_since_progress = time.monotonic() - last_progress_time
            if elapsed_since_progress > _STALL_WARNING_SECONDS and not stall_warned:
                _log.warning(
                    "FFmpeg has not reported progress for %.0fs: %s",
                    elapsed_since_progress, source,
                )
                stall_warned = True

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
            # Kill immediately to prevent stderr bleed during shutdown
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
