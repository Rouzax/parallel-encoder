"""Rich-based terminal progress display for parallel video encoding."""

from __future__ import annotations

import threading
from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from encoder.ffmpeg import EncodingResult
from encoder.media_info import format_bitrate, format_size


class EncodingProgress:
    """Thread-safe progress manager using Rich.

    Manages per-file progress bars and an overall progress bar inside a
    Rich :class:`Live` display.  Safe to call from multiple worker threads.
    """

    def __init__(self, total_files: int) -> None:
        self._lock = threading.Lock()
        self._total_files = total_files
        self._durations: dict[int, float] = {}
        self._file_progress: dict[int, float] = {}
        self._completed_count = 0

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[filename]:40.40s}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[info]}"),
            TimeRemainingColumn(),
        )

        self._overall_task = self._progress.add_task(
            "Overall",
            total=0,
            filename="Overall",
            info=f"0/{total_files} files",
        )

        self._console = Console(highlight=False)
        self._live = Live(
            self._progress,
            console=self._console,
            refresh_per_second=10,
        )

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def add_file(self, filename: str, duration: float) -> TaskID:
        """Add a new file task to the progress display.

        Args:
            filename: Display name for the file.
            duration: Total duration of the source video in seconds.

        Returns:
            A ``task_id`` that can be passed to :meth:`update_file`,
            :meth:`complete_file`, and :meth:`make_progress_callback`.
        """
        with self._lock:
            task_id: int = self._progress.add_task(
                filename,
                total=duration,
                filename=filename,
                info="waiting",
                visible=False,
            )
            self._durations[task_id] = duration
            self._file_progress[task_id] = 0.0
            current_total = self._progress.tasks[self._overall_task].total or 0
            self._progress.update(self._overall_task, total=current_total + duration)
            return task_id

    def start_file(self, task_id: TaskID) -> None:
        """Make a file bar visible when encoding begins."""
        with self._lock:
            self._progress.update(task_id, visible=True, info="encoding...")

    def update_file(self, task_id: TaskID, progress_info: dict) -> None:
        """Update the progress bar for a specific file.

        Args:
            task_id: The id returned by :meth:`add_file`.
            progress_info: Dict with at least ``time_seconds``, ``fps``,
                and ``speed`` keys.
        """
        time_seconds: float = progress_info.get("time_seconds", 0.0)
        fps: float = progress_info.get("fps", 0.0)
        speed: float = progress_info.get("speed", 0.0)
        info_text = f"{fps:.1f} fps | {speed:.2f}x"

        with self._lock:
            duration = self._durations.get(task_id, 0.0)
            completed = min(time_seconds, duration) if duration else time_seconds
            # Snap to duration when within 0.5% to avoid stale "0:00:01"
            if duration and completed >= duration * 0.995:
                completed = duration
            self._progress.update(task_id, completed=completed, info=info_text)
            self._file_progress[task_id] = completed
            self._progress.update(
                self._overall_task,
                completed=sum(self._file_progress.values()),
            )

    def complete_file(self, task_id: TaskID) -> None:
        """Mark a file task as 100 % complete and advance overall progress."""
        with self._lock:
            duration = self._durations.get(task_id, 0.0)
            self._progress.update(
                task_id,
                completed=duration,
                info="[green]done[/green]",
                visible=False,
            )
            self._file_progress[task_id] = duration
            self._completed_count += 1
            self._progress.update(
                self._overall_task,
                completed=sum(self._file_progress.values()),
                info=f"{self._completed_count}/{self._total_files} files",
            )

    def fail_file(self, task_id: TaskID) -> None:
        """Mark a file task as failed and advance overall progress."""
        with self._lock:
            duration = self._durations.get(task_id, 0.0)
            self._progress.update(
                task_id,
                completed=duration,
                info="[red]failed[/red]",
                visible=False,
            )
            self._file_progress[task_id] = duration
            self._completed_count += 1
            self._progress.update(
                self._overall_task,
                completed=sum(self._file_progress.values()),
                info=f"{self._completed_count}/{self._total_files} files",
            )

    def make_progress_callback(self, task_id: TaskID) -> Callable[[dict], None]:
        """Return a callback suitable for passing to :func:`run_encode`.

        The returned function calls :meth:`update_file` with the given
        *task_id* every time it is invoked.
        """

        def _callback(progress_info: dict) -> None:
            self.update_file(task_id, progress_info)

        return _callback

    # ------------------------------------------------------------------
    # Context manager / Live display control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Enter the Rich :class:`Live` display context."""
        self._live.start()

    def stop(self) -> None:
        """Exit the Rich :class:`Live` display context."""
        self._live.stop()

    def __enter__(self) -> EncodingProgress:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_summary_table(
    source_files: list[dict],
    results: list[EncodingResult],
    target_files: list[dict] | None = None,
) -> None:
    """Print a Rich table summarising encoding results.

    Args:
        source_files: Probe dicts for the original files.
        results: :class:`EncodingResult` objects (one per encode).
        target_files: Optional probe dicts for the encoded output files.
            When provided the table includes target codec, bitrate, size
            and a reduction percentage.
    """
    console = Console(highlight=False)

    table = Table(
        title="Encoding Summary",
        show_lines=True,
        expand=True,
    )

    table.add_column("Filename", style="bold")
    table.add_column("Source Codec")
    table.add_column("Source Bitrate", justify="right")
    table.add_column("Target Codec")
    table.add_column("Target Bitrate", justify="right")
    table.add_column("Reduction %", justify="right")
    table.add_column("Source Size", justify="right")
    table.add_column("Target Size", justify="right")
    table.add_column("Speed", justify="right")

    # Build look-ups keyed on full path (primary) and stem (fallback). --
    target_by_path: dict[str, dict] = {}
    target_by_stem: dict[str, dict] = {}
    if target_files:
        for tf in target_files:
            target_by_path[tf["path"]] = tf
            target_by_stem[tf["filename"]] = tf  # fallback for non-unique stems

    result_by_source: dict[str, EncodingResult] = {}
    for r in results:
        result_by_source[r.source_path] = r

    # One row per source file. -------------------------------------------
    for sf in source_files:
        filename: str = sf["filename"]
        source_codec: str = sf.get("video_codec") or "N/A"
        source_bitrate: str = format_bitrate(sf.get("video_bitrate") or sf.get("total_bitrate"))
        source_size: str = format_size(sf.get("file_size", 0))

        result = result_by_source.get(sf["path"])

        # Target info (only when target probe data is available). --------
        target_codec: str = "N/A"
        target_bitrate: str = "N/A"
        target_size: str = "N/A"
        reduction_text: Text = Text("N/A")

        # Match target by output path derived from result, or fall back to stem
        tf = None
        if result is not None:
            tf = target_by_path.get(result.output_path)
        if tf is None:
            tf = target_by_stem.get(filename)
        if tf is not None:
            target_codec = tf.get("video_codec") or "N/A"
            target_bitrate = format_bitrate(tf.get("video_bitrate") or tf.get("total_bitrate"))
            target_size = format_size(tf.get("file_size", 0))

            src_sz: int = sf.get("file_size", 0)
            tgt_sz: int = tf.get("file_size", 0)
            src_dur: float = sf.get("duration", 0.0)
            tgt_dur: float = tf.get("duration", 0.0)

            if src_sz > 0:
                # For test encodes (target much shorter than source),
                # compare bitrate instead of file size
                if tgt_dur > 0 and src_dur > 0 and tgt_dur < src_dur * 0.9:
                    src_br = sf.get("total_bitrate") or sf.get("video_bitrate") or 0
                    tgt_br = tf.get("total_bitrate") or tf.get("video_bitrate") or 0
                    if src_br > 0:
                        pct = ((tgt_br - src_br) / src_br) * 100
                    else:
                        pct = 0.0
                else:
                    pct = ((tgt_sz - src_sz) / src_sz) * 100
                colour = "green" if pct <= 0 else "red"
                reduction_text = Text(f"{pct:+.1f}%", style=colour)

        # Speed column. --------------------------------------------------
        speed_str: str = "N/A"
        if result is not None:
            if result.success:
                speed_str = _format_duration(result.encoding_time)
            else:
                speed_str = "[red]FAILED[/red]"

        table.add_row(
            filename,
            source_codec,
            source_bitrate,
            target_codec,
            target_bitrate,
            reduction_text,
            source_size,
            target_size,
            speed_str,
        )

    console.print()
    console.print(table)
