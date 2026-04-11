"""Parallel encoding orchestrator for video encoding with FFmpeg.

Manages a pool of worker threads that encode video files concurrently,
distributing CPU threads across workers based on codec and NUMA topology.
"""

from __future__ import annotations

import itertools
import logging
import os
import platform
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from encoder.ffmpeg import (
    EncodingResult, build_command, cover_art_attach_args, extract_cover_art,
    find_ffmpeg, run_encode,
)
from presets.loader import preset_to_ffmpeg_args

_log = logging.getLogger("parallel-encoder")

# Default (ideal) threads-per-worker by codec name.
_CODEC_THREADS: dict[str, int] = {
    "libx265": 16,
    "libsvtav1": 20,
    "libx264": 8,
    "libvpx-vp9": 8,
}
_DEFAULT_THREADS_PER_WORKER = 12

# Hard ceiling on useful threads per worker.  Codecs with tile-based
# parallelism (VP9) cannot exploit more threads than their tile layout
# allows — extra threads just sit idle.
_CODEC_MAX_THREADS: dict[str, int] = {
    "libvpx-vp9": 8,
}


def max_useful_threads(codec: str) -> int | None:
    """Return the max useful thread count for *codec*, or *None* if uncapped."""
    return _CODEC_MAX_THREADS.get(codec)


# ---------------------------------------------------------------------------
# NUMA / socket topology detection
# ---------------------------------------------------------------------------

@dataclass
class CpuTopology:
    """Describes the CPU topology of the system."""

    total_threads: int
    sockets: int
    cores_per_socket: int
    threads_per_core: int
    numa_nodes: int
    threads_per_numa: int

    @property
    def is_multi_socket(self) -> bool:
        return self.sockets > 1


def detect_topology() -> CpuTopology:
    """Detect CPU topology including NUMA layout.

    Works on Linux (via lscpu) and Windows (via wmic / powershell).
    Falls back to a flat single-socket topology on unknown platforms.
    """
    total_threads = os.cpu_count() or 4
    system = platform.system()

    if system == "Linux":
        return _detect_linux(total_threads)
    if system == "Windows":
        return _detect_windows(total_threads)
    return _flat_topology(total_threads)


def _detect_linux(total_threads: int) -> CpuTopology:
    """Parse lscpu for socket/NUMA info."""
    try:
        out = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5,
        ).stdout
        fields: dict[str, str] = {}
        for line in out.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                fields[key.strip()] = val.strip()

        _log.debug("lscpu raw fields: %s", fields)

        sockets = int(fields.get("Socket(s)", "1"))
        cores_per_socket = int(fields.get("Core(s) per socket", str(total_threads // max(sockets, 1))))
        threads_per_core = int(fields.get("Thread(s) per core", "1"))
        numa_nodes = int(fields.get("NUMA node(s)", str(sockets)))
    except Exception:
        return _flat_topology(total_threads)

    threads_per_numa = total_threads // max(numa_nodes, 1)
    return CpuTopology(
        total_threads=total_threads,
        sockets=sockets,
        cores_per_socket=cores_per_socket,
        threads_per_core=threads_per_core,
        numa_nodes=numa_nodes,
        threads_per_numa=threads_per_numa,
    )


def _detect_windows(total_threads: int) -> CpuTopology:
    """Use PowerShell to query socket/core info on Windows."""
    try:
        ps_cmd = (
            "Get-CimInstance Win32_Processor | "
            "Select-Object NumberOfCores, ThreadCount, SocketDesignation | "
            "ConvertTo-Json"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=10,
        ).stdout

        import json
        data = json.loads(out)
        _log.debug("Win32_Processor data: %s", data)
        # data may be a single object or a list (one per socket)
        if isinstance(data, dict):
            data = [data]

        sockets = len(data)
        cores_per_socket = int(data[0].get("NumberOfCores", total_threads // sockets))
        threads_per_socket = int(data[0].get("ThreadCount", cores_per_socket * 2))
        threads_per_core = threads_per_socket // max(cores_per_socket, 1)
    except Exception:
        return _flat_topology(total_threads)

    # On Windows, NUMA nodes typically map 1:1 to sockets
    numa_nodes = sockets
    threads_per_numa = total_threads // max(numa_nodes, 1)
    return CpuTopology(
        total_threads=total_threads,
        sockets=sockets,
        cores_per_socket=cores_per_socket,
        threads_per_core=threads_per_core,
        numa_nodes=numa_nodes,
        threads_per_numa=threads_per_numa,
    )


def _flat_topology(total_threads: int) -> CpuTopology:
    """Fallback: assume single socket."""
    return CpuTopology(
        total_threads=total_threads,
        sockets=1,
        cores_per_socket=total_threads,
        threads_per_core=1,
        numa_nodes=1,
        threads_per_numa=total_threads,
    )


# Container extension lookup.
_CONTAINER_EXTENSION: dict[str, str] = {
    "mkv": ".mkv",
    "matroska": ".mkv",
    "mp4": ".mp4",
    "webm": ".webm",
}


@dataclass
class WorkerConfig:
    """Describes the parallel encoding configuration."""

    num_workers: int
    threads_per_worker: int
    topology: CpuTopology
    numa_strategy: str  # "pin_to_node", "spread", or "none"

    @property
    def workers_per_numa(self) -> int:
        """How many workers are assigned to each NUMA node."""
        if self.topology.numa_nodes <= 1:
            return self.num_workers
        return max(1, self.num_workers // self.topology.numa_nodes)


def auto_detect_workers(
    codec: str,
    total_threads: int | None = None,
    topology: CpuTopology | None = None,
) -> WorkerConfig:
    """Determine the optimal number of workers, NUMA-aware.

    Strategy for multi-socket systems:
    - Prefer assigning whole NUMA nodes to workers (avoids cross-socket
      memory access).
    - If the ideal threads-per-worker fits within one NUMA node, pin each
      worker to a node and pack as many workers per node as possible.
    - If the codec needs more threads than one NUMA node has, let the OS
      spread (but this is suboptimal and we warn).

    Args:
        codec: FFmpeg video codec name (e.g. ``libx265``, ``libsvtav1``).
        total_threads: Override for total CPU threads. When *None*, detected
            from topology.
        topology: Pre-detected topology. When *None*, :func:`detect_topology`
            is called.

    Returns:
        A :class:`WorkerConfig` with worker count, thread budget, and
        NUMA strategy.
    """
    if topology is None:
        topology = detect_topology()

    if total_threads is not None:
        # Override: rebuild topology with the given thread count
        topology = CpuTopology(
            total_threads=total_threads,
            sockets=topology.sockets,
            cores_per_socket=topology.cores_per_socket,
            threads_per_core=topology.threads_per_core,
            numa_nodes=topology.numa_nodes,
            threads_per_numa=total_threads // max(topology.numa_nodes, 1),
        )

    ideal_threads = _CODEC_THREADS.get(codec, _DEFAULT_THREADS_PER_WORKER)

    if topology.is_multi_socket:
        return _plan_multi_socket(ideal_threads, topology)

    # Single socket — simple division
    num_workers = max(1, topology.total_threads // ideal_threads)
    threads_per_worker = topology.total_threads // num_workers
    _log.info(
        "Auto-detected workers: %d workers x %d threads, strategy=%s (codec=%s, total_threads=%d)",
        num_workers, threads_per_worker, "none", codec, topology.total_threads,
    )
    return WorkerConfig(
        num_workers=num_workers,
        threads_per_worker=threads_per_worker,
        topology=topology,
        numa_strategy="none",
    )


def _plan_multi_socket(ideal_threads: int, topo: CpuTopology) -> WorkerConfig:
    """Plan worker layout for a multi-socket / multi-NUMA system.

    Goal: keep each worker's threads within a single NUMA node to avoid
    cross-socket memory penalties.
    """
    tpn = topo.threads_per_numa  # threads per NUMA node

    if ideal_threads <= tpn:
        # Workers fit inside a single NUMA node — pack multiple per node
        workers_per_node = max(1, tpn // ideal_threads)
        threads_per_worker = tpn // workers_per_node
        num_workers = workers_per_node * topo.numa_nodes
        _log.info(
            "Auto-detected workers: %d workers x %d threads, strategy=%s (total_threads=%d)",
            num_workers, threads_per_worker, "pin_to_node", topo.total_threads,
        )
        return WorkerConfig(
            num_workers=num_workers,
            threads_per_worker=threads_per_worker,
            topology=topo,
            numa_strategy="pin_to_node",
        )

    # Worker needs more threads than one NUMA node — let OS spread
    num_workers = max(1, topo.total_threads // ideal_threads)
    threads_per_worker = topo.total_threads // max(num_workers, 1)
    _log.info(
        "Auto-detected workers: %d workers x %d threads, strategy=%s (total_threads=%d)",
        num_workers, threads_per_worker, "spread", topo.total_threads,
    )
    return WorkerConfig(
        num_workers=num_workers,
        threads_per_worker=threads_per_worker,
        topology=topo,
        numa_strategy="spread",
    )


@dataclass
class EncodingJob:
    """Describes a single video encoding task."""

    source_path: str
    output_path: str
    preset_args: list[str]
    threads: int
    test_encode: dict | None = field(default=None)
    numa_node: int | None = field(default=None)
    cover_art: list[dict] = field(default_factory=list)
    attachment_count: int = 0


class ParallelEncoder:
    """Run multiple FFmpeg encodes in parallel using a thread pool.

    On multi-socket systems, workers are assigned to NUMA nodes in a
    round-robin fashion so their FFmpeg processes can be pinned to
    specific cores via ``numactl`` (Linux) or CPU affinity (Windows).
    """

    def __init__(
        self,
        worker_config: WorkerConfig,
        ffmpeg_path: str | None = None,
    ) -> None:
        self.config = worker_config
        self.num_workers = worker_config.num_workers
        self.threads_per_worker = worker_config.threads_per_worker
        self.ffmpeg_path = ffmpeg_path if ffmpeg_path is not None else find_ffmpeg()
        self._cancel_event = threading.Event()
        import shutil as _shutil
        self._numactl_available: bool = _shutil.which("numactl") is not None
        # NUMA node assignment: each worker thread gets a fixed node on
        # first use via round-robin, and keeps it for all subsequent jobs.
        self._worker_numa: dict[int, int] = {}
        self._worker_numa_lock = threading.Lock()
        self._numa_counter = itertools.count()
        # Track active FFmpeg subprocesses so they can be killed on Ctrl+C.
        self._active_processes: list[subprocess.Popen] = []  # type: ignore[type-arg]
        self._active_processes_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Job preparation
    # ------------------------------------------------------------------

    def prepare_jobs(
        self,
        source_files: list[dict],
        source_folder: str,
        output_folder: str,
        preset: dict,
        test_encode: bool = False,
        test_seconds: int = 120,
        overwrite: bool = False,
    ) -> tuple[list[EncodingJob], list[str]]:
        """Build a list of :class:`EncodingJob` from probed source files.

        Args:
            source_files: List of dicts as returned by
                :func:`encoder.media_info.probe_folder`.
            source_folder: Root folder that the source files were scanned from.
            output_folder: Root folder where encoded files will be written.
            preset: A single preset configuration dict.
            test_encode: When *True*, each job encodes only a short segment
                from the middle of the source video.
            test_seconds: Maximum duration (in seconds) for test encodes.
            overwrite: When *True*, re-encode even if output file exists.

        Returns:
            Tuple of (list of :class:`EncodingJob` ready for :meth:`run`,
            list of skipped source file paths).
        """
        source_root = Path(source_folder).resolve()
        output_root = Path(output_folder).resolve()

        # Determine output extension from preset container.
        container: str = preset.get("container", "mkv")
        extension = _CONTAINER_EXTENSION.get(container.lower(), ".mkv")

        jobs: list[EncodingJob] = []
        skipped: list[str] = []
        for source_info in source_files:
            source_path = Path(source_info["path"])

            # Preserve folder structure relative to source_folder.
            try:
                relative = source_path.relative_to(source_root)
            except ValueError:
                relative = Path(source_path.name)

            output_path = output_root / relative.with_suffix(extension)

            # Skip files whose output already exists (unless overwriting or test-encoding).
            if not overwrite and not test_encode and output_path.exists():
                _log.info("Skipping %s (output exists: %s)", source_path.name, output_path)
                skipped.append(str(source_path))
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)

            preset_args = preset_to_ffmpeg_args(preset, source_info)

            test_encode_dict: dict | None = None
            if test_encode:
                duration: float = source_info.get("duration", 0.0)
                if duration > test_seconds:
                    start = (duration - test_seconds) / 2
                    test_encode_dict = {
                        "start_seconds": round(start, 2),
                        "duration_seconds": test_seconds,
                    }
                # If video is shorter than test_seconds, encode the whole thing
                # (test_encode_dict stays None).

            # Carry cover art info for MKV re-attachment (WebM doesn't
            # support attachments; sidecar files handle external artwork)
            cover_art: list[dict] = source_info.get("cover_art", [])

            jobs.append(
                EncodingJob(
                    source_path=str(source_path),
                    output_path=str(output_path),
                    preset_args=preset_args,
                    threads=self.threads_per_worker,
                    test_encode=test_encode_dict,
                    cover_art=cover_art if container.lower() in ("mkv", "matroska") else [],
                    attachment_count=source_info.get("attachment_count", 0),
                )
            )

        # Check for output path collisions
        seen_outputs: dict[str, str] = {}  # output_path -> source_path
        for job in jobs:
            if job.output_path in seen_outputs:
                raise ValueError(
                    f"Output path collision: both '{seen_outputs[job.output_path]}' "
                    f"and '{job.source_path}' would write to '{job.output_path}'"
                )
            seen_outputs[job.output_path] = job.source_path

        return jobs, skipped

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        jobs: list[EncodingJob],
        progress_callback: Callable[[str, dict], None] | None = None,
        completion_callback: Callable[[EncodingResult], None] | None = None,
        start_callback: Callable[[str], None] | None = None,
    ) -> list[EncodingResult]:
        """Execute encoding jobs in parallel.

        Args:
            jobs: List of :class:`EncodingJob` to process.
            progress_callback: Optional callable invoked as
                ``progress_callback(filename, progress_dict)`` for real-time
                per-file progress reporting.
            completion_callback: Optional callable invoked with each
                :class:`EncodingResult` as soon as a file finishes.
            start_callback: Optional callable invoked with the filename
                when a worker begins encoding a file.

        Returns:
            List of :class:`EncodingResult`, one per job (order may differ
            from the input list).
        """
        results: list[EncodingResult] = []
        futures: dict[Future[EncodingResult], EncodingJob] = {}
        processed: set[Future[EncodingResult]] = set()
        self._worker_numa.clear()

        import signal
        is_main_thread = threading.current_thread() is threading.main_thread()
        original_handler = signal.getsignal(signal.SIGINT) if is_main_thread else None

        def _sigint_handler(signum: int, frame: object) -> None:
            """Handle Ctrl+C immediately rather than waiting for KeyboardInterrupt."""
            self._cancel_event.set()
            with self._active_processes_lock:
                for proc in self._active_processes:
                    try:
                        proc.kill()
                    except OSError:
                        pass
            _log.info("Received interrupt, stopping all encodes...")

        try:
            if is_main_thread:
                signal.signal(signal.SIGINT, _sigint_handler)
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for job in jobs:
                    future = executor.submit(self._run_single, job, progress_callback, start_callback)
                    futures[future] = job

                for future in as_completed(futures):
                    processed.add(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        job = futures[future]
                        _log.error("Encoding crashed: %s — %s", job.source_path, exc)
                        result = EncodingResult(
                            source_path=job.source_path,
                            output_path=job.output_path,
                            success=False,
                            exit_code=-1,
                            encoding_time=0.0,
                            error_message=f"Internal error: {exc}",
                        )
                    results.append(result)
                    if completion_callback is not None:
                        completion_callback(result)

        except KeyboardInterrupt:
            # Signal handler already set cancel_event and killed processes,
            # but handle the case where KeyboardInterrupt arrives first.
            self._cancel_event.set()
            for future in futures:
                future.cancel()

        finally:
            if is_main_thread:
                signal.signal(signal.SIGINT, original_handler)

        # Recover any results from futures that completed but were not yet
        # iterated by the as_completed loop (e.g. when KeyboardInterrupt
        # broke out of it early).  Skip futures the main loop already
        # processed so we never double-count.
        for future in futures:
            if future in processed:
                continue
            if future.done() and not future.cancelled():
                try:
                    results.append(future.result())
                except Exception:
                    pass

        return results

    def _get_worker_numa(self) -> int | None:
        """Return the NUMA node for the current worker thread.

        On first call from a given thread, assigns a node via round-robin
        and caches it.  Returns None when NUMA pinning is not active.
        """
        if self.config.numa_strategy != "pin_to_node":
            return None
        tid = threading.get_ident()
        node = self._worker_numa.get(tid)
        if node is not None:
            return node
        with self._worker_numa_lock:
            # Double-check after acquiring lock
            if tid in self._worker_numa:
                return self._worker_numa[tid]
            node = next(self._numa_counter) % self.config.topology.numa_nodes
            self._worker_numa[tid] = node
            _log.debug("Worker thread %d assigned to NUMA node %d", tid, node)
            return node

    def _has_numactl(self) -> bool:
        """Check if numactl is available (Linux only)."""
        return self._numactl_available

    def _wrap_numa(self, command: list[str], numa_node: int | None) -> list[str]:
        """Wrap a command with NUMA affinity if applicable.

        On Linux with numactl available:
            numactl --cpunodebind=N --membind=N ffmpeg ...
        This pins the FFmpeg process and all its threads to cores and memory
        on the specified NUMA node, eliminating cross-socket penalties.

        On Windows, CPU affinity is set via the subprocess start info
        (handled in run_encode_with_affinity).
        """
        if numa_node is None:
            return command
        system = platform.system()
        if system == "Linux" and self._has_numactl():
            return [
                "numactl",
                f"--cpunodebind={numa_node}",
                f"--membind={numa_node}",
            ] + command
        # On Windows or without numactl, return unmodified.
        # x265/SVT-AV1 have their own NUMA awareness that helps somewhat.
        return command

    def _run_single(
        self,
        job: EncodingJob,
        progress_callback: Callable[[str, dict], None] | None,
        start_callback: Callable[[str], None] | None = None,
    ) -> EncodingResult:
        """Execute a single encoding job (runs inside a worker thread)."""
        if self._cancel_event.is_set():
            return EncodingResult(
                source_path=job.source_path,
                output_path=job.output_path,
                success=False,
                exit_code=-1,
                encoding_time=0.0,
                error_message="Encoding cancelled.",
            )

        cover_art_temp_dir: str | None = None
        try:
            # Extract cover art to temp files and build -attach args
            attach_args: list[str] = []
            if job.cover_art:
                import tempfile
                cover_art_temp_dir = tempfile.mkdtemp(prefix="pe_cover_")
                extracted = extract_cover_art(
                    self.ffmpeg_path, job.source_path, job.cover_art, cover_art_temp_dir,
                )
                attach_args = cover_art_attach_args(
                    extracted,
                    existing_attachment_count=job.attachment_count,
                )

            command = build_command(
                ffmpeg_path=self.ffmpeg_path,
                source=job.source_path,
                output=job.output_path,
                preset_args=job.preset_args + attach_args,
                threads=job.threads,
                test_encode=job.test_encode,
            )

            # Pin to NUMA node based on worker thread, not job
            numa_node = self._get_worker_numa()
            command = self._wrap_numa(command, numa_node)

            filename = Path(job.source_path).name

            # Signal that this worker has started encoding
            if start_callback is not None:
                start_callback(filename)

            per_file_cb: Callable[[dict], None] | None = None
            if progress_callback is not None:
                def _make_cb(fn: str = filename) -> Callable[[dict], None]:
                    def _cb(progress: dict) -> None:
                        progress_callback(fn, progress)
                    return _cb
                per_file_cb = _make_cb()

            def _on_process_started(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
                with self._active_processes_lock:
                    self._active_processes.append(proc)

            def _on_process_ended(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
                with self._active_processes_lock:
                    try:
                        self._active_processes.remove(proc)
                    except ValueError:
                        pass

            result = run_encode(
                command,
                progress_callback=per_file_cb,
                cancel_event=self._cancel_event,
                numa_node=numa_node,
                threads_per_numa=self.config.topology.threads_per_numa,
                process_started=_on_process_started,
                process_ended=_on_process_ended,
            )

            return result
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            _log.error("Unexpected error encoding %s: %s", job.source_path, exc, exc_info=True)
            # Clean up temp file that may have been left behind
            from encoder.ffmpeg import atomic_output_path, cleanup_temp
            cleanup_temp(atomic_output_path(job.output_path))
            return EncodingResult(
                source_path=job.source_path,
                output_path=job.output_path,
                success=False,
                exit_code=-1,
                encoding_time=0.0,
                error_message=f"Internal error: {exc}",
            )
        finally:
            # Clean up the entire cover art temp directory
            if cover_art_temp_dir is not None:
                import shutil
                shutil.rmtree(cover_art_temp_dir, ignore_errors=True)
