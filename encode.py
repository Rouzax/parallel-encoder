"""Parallel video encoder CLI — encode video files using FFmpeg with parallel workers."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any

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


def _cleanup_test_outputs(output_paths: list[str]) -> None:
    """Remove only the files that the test encode created."""
    for path_str in output_paths:
        try:
            Path(path_str).unlink()
        except FileNotFoundError:
            pass


def _copy_non_video_files(
    source_folder: Path,
    output_folder: Path,
    video_extensions: tuple[str, ...],
) -> int:
    """Copy non-video files from source to output preserving folder structure.

    Returns the number of files copied.
    """
    copied = 0
    for src_file in source_folder.rglob("*"):
        if not src_file.is_file() or src_file.is_symlink():
            continue
        if src_file.suffix.lstrip(".").lower() in video_extensions:
            continue
        relative = src_file.relative_to(source_folder)
        dst_file = output_folder / relative
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
) -> list[EncodingResult]:
    """Core encoding routine shared by test and full encode paths."""
    encoder = ParallelEncoder(
        worker_config=worker_config,
        ffmpeg_path=ffmpeg_path,
    )
    num_workers = worker_config.num_workers
    threads_per_worker = worker_config.threads_per_worker

    jobs = encoder.prepare_jobs(
        source_files=source_files,
        source_folder=source_folder,
        output_folder=output_folder,
        preset=preset,
        test_encode=test_encode,
        test_seconds=test_seconds,
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
        return []

    mode = "test encode" if test_encode else "full encode"
    console.print(
        f"\n[bold]Starting {mode}:[/bold] {len(jobs)} file(s), "
        f"{num_workers} worker(s) x {threads_per_worker} threads, "
        f"preset [cyan]{preset_key}[/cyan]\n"
    )

    with EncodingProgress(total_files=len(jobs)) as progress:
        # Register files with the progress display.
        task_ids: dict[str, Any] = {}
        for job in jobs:
            filename = Path(job.source_path).name
            duration = next(
                (sf["duration"] for sf in source_files if sf["path"] == job.source_path),
                0.0,
            )
            if test_encode and job.test_encode:
                duration = job.test_encode["duration_seconds"]
            task_id = progress.add_file(filename, duration)
            task_ids[filename] = task_id

        def on_progress(filename: str, info: dict) -> None:
            tid = task_ids.get(filename)
            if tid is not None:
                progress.update_file(tid, info)

        results = encoder.run(jobs, progress_callback=on_progress)

        # Mark completed files.
        for result in results:
            fname = Path(result.source_path).name
            tid = task_ids.get(fname)
            if tid is not None:
                progress.complete_file(tid)

    return results


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
            test_results = _run_encoding(
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
    results = _run_encoding(
        source_files=source_files,
        source_folder=source,
        output_folder=output,
        preset_key=preset_key,
        preset=preset_cfg,
        worker_config=worker_cfg,
        ffmpeg_path=ffmpeg_path,
        dry_run=dry_run,
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

    # ── Copy non-video files ────────────────────────────────────────
    if copy_all:
        log.info("Copying non-video files...")
        count = _copy_non_video_files(Path(source), Path(output), video_extensions)
        log.info("Copied %d non-video file(s).", count)

    successful = sum(1 for r in results if r.success)
    console.print(f"\n[bold green]Done![/bold green] {successful}/{len(results)} file(s) encoded successfully.")


if __name__ == "__main__":
    main()
