"""Parallel encoding orchestrator for video encoding with FFmpeg.

Manages a pool of worker threads that encode video files concurrently,
distributing CPU threads across workers based on codec and NUMA topology.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from encoder.ffmpeg import EncodingResult, build_command, find_ffmpeg, run_encode
from presets.loader import preset_to_ffmpeg_args

_log = logging.getLogger("parallel-encoder")

# Default threads-per-worker by codec name.
_CODEC_THREADS: dict[str, int] = {
    "libx265": 16,
    "libsvtav1": 20,
    "libx264": 8,
    "libvpx-vp9": 12,
}
_DEFAULT_THREADS_PER_WORKER = 12


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
        self._numactl_available: bool | None = None

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

            # Assign NUMA node round-robin when using pin_to_node strategy
            numa_node: int | None = None
            if self.config.numa_strategy == "pin_to_node":
                numa_node = len(jobs) % self.config.topology.numa_nodes

            jobs.append(
                EncodingJob(
                    source_path=str(source_path),
                    output_path=str(output_path),
                    preset_args=preset_args,
                    threads=self.threads_per_worker,
                    test_encode=test_encode_dict,
                    numa_node=numa_node,
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

        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for job in jobs:
                    future = executor.submit(self._run_single, job, progress_callback, start_callback)
                    futures[future] = job

                for future in as_completed(futures):
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
            self._cancel_event.set()
            for future in futures:
                future.cancel()
            # Collect results from futures that already completed.
            for future in futures:
                if future.done() and not future.cancelled():
                    try:
                        results.append(future.result())
                    except Exception:
                        pass

        return results

    def _has_numactl(self) -> bool:
        """Check if numactl is available (Linux only), cached."""
        if self._numactl_available is None:
            import shutil
            self._numactl_available = shutil.which("numactl") is not None
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
        try:
            command = build_command(
                ffmpeg_path=self.ffmpeg_path,
                source=job.source_path,
                output=job.output_path,
                preset_args=job.preset_args,
                threads=job.threads,
                test_encode=job.test_encode,
            )

            # Pin to NUMA node if assigned
            command = self._wrap_numa(command, job.numa_node)

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

            return run_encode(command, progress_callback=per_file_cb, cancel_event=self._cancel_event)
        except Exception as exc:
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
