"""Parallel video encoder CLI — encode video files using FFmpeg with parallel workers."""

from __future__ import annotations

import logging
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable

import click
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from encoder.ffmpeg import EncodingResult, find_ffmpeg, find_ffprobe  # noqa: E402
from encoder.media_info import probe_folder
from encoder.worker_pool import (
    ParallelEncoder,
    WorkerConfig,
    auto_detect_workers,
    detect_topology,
)
from presets.loader import get_preset_by_name, list_preset_names, load_presets
from ui.progress import EncodingProgress, print_summary_table

console = Console(highlight=False)

# Default preset file location: config/presets.yaml next to this script.
_DEFAULT_PRESET_FILE = Path(__file__).resolve().parent / "config" / "presets.yaml"


def _select_preset_interactive(presets: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Show an interactive menu and return (key, config) for the chosen preset."""
    names = list_preset_names(presets)

    if len(names) == 1:
        console.print(f"[cyan]Auto-selecting the only preset:[/cyan] {names[0]}")
        return get_preset_by_name(presets, names[0])

    console.print("\n[bold]Available presets:[/bold]")
    for i, name in enumerate(names, 1):
        console.print(f"  [cyan]{i:>2}[/cyan]  {name}")

    while True:
        choice = Prompt.ask(
            "\nSelect a preset by number",
            console=console,
        )
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                return get_preset_by_name(presets, names[idx])
        except ValueError:
            pass
        console.print("[red]Invalid selection, try again.[/red]")


def _run_vmaf_scoring(
    ffmpeg_path: str,
    source_files: list[dict[str, Any]],
    test_results: list[EncodingResult],
    test_seconds: int,
) -> None:
    """Run VMAF scoring on test encode results."""
    from encoder.vmaf import check_vmaf_support, run_vmaf, vmaf_quality_label
    from rich.table import Table

    if not check_vmaf_support(ffmpeg_path):
        console.print("[red]VMAF not available: your ffmpeg build does not include libvmaf.[/red]")
        return

    console.print("\n[bold]Running VMAF quality scoring...[/bold]")

    vmaf_table = Table(title="VMAF Quality Scores", show_lines=True)
    vmaf_table.add_column("Filename")
    vmaf_table.add_column("VMAF Score", justify="right")
    vmaf_table.add_column("Quality")

    for result in test_results:
        if not result.success:
            vmaf_table.add_row(Path(result.source_path).name, "N/A", "encode failed")
            continue

        # Find the matching source info
        src_info = next(
            (sf for sf in source_files if sf["path"] == result.source_path),
            None,
        )
        if src_info is None:
            continue

        duration = src_info.get("duration", 0.0)
        start_seconds = None
        dur_seconds = None
        if duration > test_seconds:
            start_seconds = (duration - test_seconds) / 2
            dur_seconds = test_seconds

        # Get source and target dimensions
        src_w = src_info.get("video_width", 1280)
        src_h = src_info.get("video_height", 720)

        # Probe the encoded file for its actual resolution
        from encoder.media_info import probe_file
        try:
            target_info = probe_file(result.output_path)
            tgt_w = target_info.get("video_width", src_w)
            tgt_h = target_info.get("video_height", src_h)
        except RuntimeError:
            tgt_w, tgt_h = 1280, 720

        console.print(f"  Scoring [cyan]{Path(result.source_path).name}[/cyan]...")

        scores = run_vmaf(
            ffmpeg_path=ffmpeg_path,
            source_path=result.source_path,
            encoded_path=result.output_path,
            source_width=src_w,
            source_height=src_h,
            target_width=tgt_w,
            target_height=tgt_h,
            start_seconds=start_seconds,
            duration_seconds=dur_seconds,
        )

        if scores is not None:
            vmaf_score = scores["vmaf"]
            label = vmaf_quality_label(vmaf_score)
            color = "green" if vmaf_score >= 80 else "yellow" if vmaf_score >= 70 else "red"
            vmaf_table.add_row(
                Path(result.source_path).name,
                f"[{color}]{vmaf_score:.1f}[/{color}]",
                label,
            )
        else:
            vmaf_table.add_row(Path(result.source_path).name, "N/A", "scoring failed")

    console.print()
    console.print(vmaf_table)


def _cleanup_test_outputs(output_paths: list[str]) -> None:
    """Remove only the files that the test encode created."""
    for path_str in output_paths:
        try:
            Path(path_str).unlink()
        except FileNotFoundError:
            pass


def _copy_sidecars_for_file(
    source_path: Path,
    source_root: Path,
    output_root: Path,
    video_extensions: tuple[str, ...],
) -> int:
    """Copy non-video sidecar files from the same directory as *source_path*.

    Only copies files that don't already exist in the output (avoids
    re-copying when multiple videos share a directory).

    Returns the number of files copied.
    """
    ext_lower: set[str] = {e.lower().lstrip(".") for e in video_extensions}
    src_dir = source_path.parent
    copied = 0
    for src_file in src_dir.iterdir():
        if not src_file.is_file() or src_file.is_symlink():
            continue
        if src_file.suffix.lstrip(".").lower() in ext_lower:
            continue
        relative = src_file.relative_to(source_root)
        dst_file = output_root / relative
        if dst_file.exists():
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1
    return copied


def _run_encoding(
    source_files: list[dict[str, Any]],
    source_folder: str,
    output_folder: str,
    preset_key: str,
    preset: dict[str, Any],
    worker_config: WorkerConfig,
    ffmpeg_path: str,
    test_encode: bool = False,
    test_seconds: int = 120,
    dry_run: bool = False,
    overwrite: bool = False,
    on_file_complete: Callable[[EncodingResult], None] | None = None,
) -> tuple[list[EncodingResult], list[str]]:
    """Core encoding routine shared by test and full encode paths.

    Returns:
        Tuple of (encoding results, list of skipped source paths).
    """
    encoder = ParallelEncoder(
        worker_config=worker_config,
        ffmpeg_path=ffmpeg_path,
    )
    num_workers = worker_config.num_workers
    threads_per_worker = worker_config.threads_per_worker

    jobs, skipped = encoder.prepare_jobs(
        source_files=source_files,
        source_folder=source_folder,
        output_folder=output_folder,
        preset=preset,
        test_encode=test_encode,
        test_seconds=test_seconds,
        overwrite=overwrite,
    )

    if skipped:
        console.print(
            f"[yellow]Skipped {len(skipped)} file(s) (output exists).[/yellow] "
            f"Use [bold]--overwrite[/bold] to re-encode."
        )

    _log = logging.getLogger("parallel-encoder")
    for job in jobs:
        _log.debug("Prepared job: source=%s output=%s threads=%d numa=%s",
                    job.source_path, job.output_path, job.threads, job.numa_node)

    if dry_run:
        from encoder.ffmpeg import build_command

        console.print("\n[bold]Dry run — commands that would be executed:[/bold]\n")
        for job in jobs:
            cmd = build_command(
                ffmpeg_path=ffmpeg_path,
                source=job.source_path,
                output=job.output_path,
                preset_args=job.preset_args,
                threads=job.threads,
                test_encode=job.test_encode,
            )
            console.print(f"[dim]{' '.join(cmd)}[/dim]\n")
        return [], skipped

    if not jobs:
        return [], skipped

    mode = "test encode" if test_encode else "full encode"
    console.print(
        f"\n[bold]Starting {mode}:[/bold] {len(jobs)} file(s), "
        f"{num_workers} worker(s) x {threads_per_worker} threads, "
        f"preset [cyan]{preset_key}[/cyan]\n"
    )

    with EncodingProgress(total_files=len(jobs)) as progress:
        # Register files with the progress display.
        # Normalize filenames (NFC) to avoid mismatches with accented
        # characters that may be stored in decomposed form (NFD) on disk.
        task_ids: dict[str, Any] = {}
        for job in jobs:
            filename = Path(job.source_path).name
            key = unicodedata.normalize("NFC", filename)
            duration = next(
                (sf["duration"] for sf in source_files if sf["path"] == job.source_path),
                0.0,
            )
            if test_encode and job.test_encode:
                duration = job.test_encode["duration_seconds"]
            task_id = progress.add_file(filename, duration)
            task_ids[key] = task_id

        def on_start(filename: str) -> None:
            key = unicodedata.normalize("NFC", filename)
            tid = task_ids.get(key)
            if tid is not None:
                progress.start_file(tid)

        def on_progress(filename: str, info: dict) -> None:
            key = unicodedata.normalize("NFC", filename)
            tid = task_ids.get(key)
            if tid is not None:
                progress.update_file(tid, info)

        def _on_complete(result: EncodingResult) -> None:
            fname = Path(result.source_path).name
            key = unicodedata.normalize("NFC", fname)
            tid = task_ids.get(key)
            if tid is not None:
                if result.success:
                    progress.complete_file(tid)
                else:
                    progress.fail_file(tid)
            if on_file_complete is not None and result.success:
                on_file_complete(result)

        results = encoder.run(
            jobs,
            progress_callback=on_progress,
            completion_callback=_on_complete,
            start_callback=on_start,
        )

    return results, skipped


@click.command()
@click.option(
    "--source", "-s",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Source folder containing video files.",
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(file_okay=False, resolve_path=True),
    help="Output folder for encoded files.",
)
@click.option(
    "--preset", "-p",
    default=None,
    help="Preset display name. If omitted, an interactive menu is shown.",
)
@click.option(
    "--preset-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to presets YAML file. Defaults to config/presets.yaml.",
)
@click.option(
    "--workers", "-w",
    default=None,
    type=int,
    help="Number of parallel workers. Auto-detected if omitted.",
)
@click.option("--test-encode", is_flag=True, help="Run a test encode before full encode.")
@click.option("--test-only", is_flag=True, help="Run test encode and exit (no prompt, no full encode).")
@click.option("--test-seconds", default=120, type=int, help="Duration of test encode in seconds.")
@click.option("--copy-all", is_flag=True, help="Copy non-video files to the output folder.")
@click.option("--dry-run", is_flag=True, help="Print FFmpeg commands without executing.")
@click.option("--overwrite", is_flag=True, help="Re-encode files even if output already exists.")
@click.option("--vmaf", is_flag=True, help="Run VMAF quality scoring after test encode (requires --test-only or --test-encode).")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v info, -vv debug).")
@click.option("--log-file", default=None, type=click.Path(dir_okay=False), help="Write debug log to file.")
def main(
    source: str,
    output: str,
    preset: str | None,
    preset_file: str | None,
    workers: int | None,
    test_encode: bool,
    test_only: bool,
    test_seconds: int,
    copy_all: bool,
    dry_run: bool,
    overwrite: bool,
    vmaf: bool,
    verbose: int,
    log_file: str | None,
) -> None:
    """Parallel video encoder using FFmpeg.

    Encodes video files from SOURCE folder to OUTPUT folder using the selected
    preset, running multiple FFmpeg processes in parallel.
    """
    from logger import setup_logging
    log = setup_logging(verbosity=verbose, log_file=log_file)

    # ── Check dependencies ──────────────────────────────────────────
    try:
        ffmpeg_path = find_ffmpeg()
        find_ffprobe()
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    log.info("Using ffmpeg: %s", ffmpeg_path)

    # ── Load presets ────────────────────────────────────────────────
    pf = Path(preset_file) if preset_file else _DEFAULT_PRESET_FILE
    try:
        presets = load_presets(pf)
    except (FileNotFoundError, KeyError) as e:
        console.print(f"[red]Error loading presets:[/red] {e}")
        sys.exit(1)

    log.info("Loaded %d preset(s) from %s", len(presets), pf)

    # ── Select preset ───────────────────────────────────────────────
    if preset:
        try:
            preset_key, preset_cfg = get_preset_by_name(presets, preset)
        except ValueError:
            console.print(f"[red]Preset not found:[/red] '{preset}'")
            console.print("[dim]Available presets:[/dim]")
            for name in list_preset_names(presets):
                console.print(f"  - {name}")
            sys.exit(1)
    else:
        preset_key, preset_cfg = _select_preset_interactive(presets)

    console.print(f"[bold]Preset:[/bold] [cyan]{preset_cfg['display_name']}[/cyan]")

    # ── Detect topology and determine worker count ───────────────────
    codec = preset_cfg["video"]["codec"]
    topo = detect_topology()

    log.info("CPU: %d socket(s), %d cores/socket, %d threads/core = %d threads, %d NUMA node(s)",
             topo.sockets, topo.cores_per_socket, topo.threads_per_core, topo.total_threads, topo.numa_nodes)

    if workers:
        worker_cfg = WorkerConfig(
            num_workers=workers,
            threads_per_worker=max(1, topo.total_threads // workers),
            topology=topo,
            numa_strategy="pin_to_node" if topo.is_multi_socket else "none",
        )
    else:
        worker_cfg = auto_detect_workers(codec, topology=topo)

    numa_info = ""
    if worker_cfg.numa_strategy == "pin_to_node":
        numa_info = f", NUMA pinning: {worker_cfg.workers_per_numa} worker(s)/node"
    elif worker_cfg.numa_strategy == "spread":
        numa_info = ", NUMA: spread (worker needs more threads than one node)"

    log.info("Workers: %d, threads/worker: %d%s", worker_cfg.num_workers, worker_cfg.threads_per_worker, numa_info)

    # ── Create output directory ─────────────────────────────────────
    Path(output).mkdir(parents=True, exist_ok=True)

    # ── Clean up stale temp files from previous runs ───────────────
    # atomic_output_path() creates files like "video.tmp.mkv"
    # atomic_output_path() produces "name.tmp.ext" — match exactly that pattern
    _TEMP_EXTENSIONS = (".mkv", ".mp4", ".webm")
    stale_temps: list[Path] = []
    for ext in _TEMP_EXTENSIONS:
        stale_temps.extend(p for p in Path(output).rglob(f"*.tmp{ext}") if p.is_file())
    for tmp in stale_temps:
        log.info("Removing stale temp file: %s", tmp)
        try:
            tmp.unlink()
        except OSError as exc:
            log.warning("Could not remove stale temp file %s: %s", tmp, exc)
    if stale_temps:
        console.print(
            f"[dim]Removed {len(stale_temps)} stale temp file(s) from previous run.[/dim]"
        )

    # ── Scan source files ───────────────────────────────────────────
    video_extensions = ("mp4", "m4v", "mkv", "avi", "mov", "wmv", "ts", "flv", "webm", "mpeg", "mpg")
    console.print(f"\n[bold]Scanning source folder:[/bold] {source}")
    source_files = probe_folder(source, extensions=video_extensions)

    if not source_files:
        console.print("[red]No video files found in source folder.[/red]")
        sys.exit(1)

    console.print(f"Found [cyan]{len(source_files)}[/cyan] video file(s).\n")

    # ── Show source info table ──────────────────────────────────────
    src_table = Table(title="Source Files", show_lines=True)
    src_table.add_column("Filename")
    src_table.add_column("Codec")
    src_table.add_column("Resolution")
    src_table.add_column("Bitrate", justify="right")
    src_table.add_column("Duration", justify="right")
    from encoder.media_info import format_bitrate
    for sf in source_files:
        mins, secs = divmod(int(sf.get("duration", 0)), 60)
        hours, mins = divmod(mins, 60)
        dur_str = f"{hours}:{mins:02d}:{secs:02d}" if hours else f"{mins}:{secs:02d}"
        src_table.add_row(
            sf["filename"],
            sf.get("video_codec", "N/A"),
            f"{sf.get('video_width', '?')}x{sf.get('video_height', '?')}",
            format_bitrate(sf.get("total_bitrate")),
            dur_str,
        )
    console.print(src_table)

    # ── Test encode workflow ────────────────────────────────────────
    if test_only:
        test_encode = True
    if test_encode and not dry_run:
        while True:
            console.print("\n[bold yellow]Running test encode...[/bold yellow]")
            test_results, _ = _run_encoding(
                source_files=source_files,
                source_folder=source,
                output_folder=output,
                preset_key=preset_key,
                preset=preset_cfg,
                worker_config=worker_cfg,
                ffmpeg_path=ffmpeg_path,
                test_encode=True,
                test_seconds=test_seconds,
            )

            # Probe test outputs for comparison.
            test_target_files = probe_folder(output, extensions=video_extensions)
            print_summary_table(source_files, test_results, test_target_files)

            if test_only:
                # Run VMAF scoring if requested
                if vmaf:
                    _run_vmaf_scoring(
                        ffmpeg_path=ffmpeg_path,
                        source_files=source_files,
                        test_results=test_results,
                        test_seconds=test_seconds,
                    )

                # Keep the test output files and exit
                successful = sum(1 for r in test_results if r.success)
                console.print(f"\n[bold green]Test encode done.[/bold green] {successful}/{len(test_results)} file(s) encoded.")
                sys.exit(0)

            proceed = Prompt.ask(
                "\nIs the bitrate acceptable?",
                choices=["y", "n", "change"],
                default="y",
                console=console,
            )

            # Clean up only the test output files (not the entire directory)
            test_output_paths = [r.output_path for r in test_results]
            _cleanup_test_outputs(test_output_paths)

            if proceed == "y":
                break
            elif proceed == "change":
                preset_key, preset_cfg = _select_preset_interactive(presets)
                codec = preset_cfg["video"]["codec"]
                if not workers:
                    worker_cfg = auto_detect_workers(codec, topology=topo)
                console.print(f"[bold]New preset:[/bold] [cyan]{preset_cfg['display_name']}[/cyan]")
            else:
                console.print("[yellow]Aborting.[/yellow]")
                sys.exit(0)

    # ── Full encode ─────────────────────────────────────────────────
    sidecar_cb: Callable[[EncodingResult], None] | None = None
    if copy_all:
        source_root = Path(source)
        output_root = Path(output)

        def _copy_sidecars(result: EncodingResult) -> None:
            _copy_sidecars_for_file(
                Path(result.source_path), source_root, output_root, video_extensions,
            )

        sidecar_cb = _copy_sidecars

    results, skipped_paths = _run_encoding(
        source_files=source_files,
        source_folder=source,
        output_folder=output,
        preset_key=preset_key,
        preset=preset_cfg,
        worker_config=worker_cfg,
        ffmpeg_path=ffmpeg_path,
        dry_run=dry_run,
        overwrite=overwrite,
        on_file_complete=sidecar_cb,
    )

    # Copy sidecars for skipped files too (their directories may have new sidecars).
    if copy_all and skipped_paths:
        source_root = Path(source)
        output_root = Path(output)
        for sp in skipped_paths:
            _copy_sidecars_for_file(
                Path(sp), source_root, output_root, video_extensions,
            )

    if dry_run:
        sys.exit(0)

    # ── Summary ─────────────────────────────────────────────────────
    target_files = probe_folder(output, extensions=video_extensions)
    print_summary_table(source_files, results, target_files)

    failed = [r for r in results if not r.success]
    if failed:
        console.print(f"\n[red]{len(failed)} file(s) failed to encode:[/red]")
        for r in failed:
            console.print(f"  [red]-[/red] {Path(r.source_path).name}: {r.error_message or 'unknown error'}")

    successful = sum(1 for r in results if r.success)
    parts = [f"{successful}/{len(results)} file(s) encoded successfully"]
    if skipped_paths:
        parts.append(f"{len(skipped_paths)} skipped")
    console.print(f"\n[bold green]Done![/bold green] {', '.join(parts)}.")


if __name__ == "__main__":
    main()
