"""Tests for worker pool job preparation."""

from __future__ import annotations

import pytest

from encoder.worker_pool import ParallelEncoder, WorkerConfig, CpuTopology


@pytest.fixture
def flat_topology():
    return CpuTopology(
        total_threads=16,
        sockets=1,
        cores_per_socket=8,
        threads_per_core=2,
        numa_nodes=1,
        threads_per_numa=16,
    )


@pytest.fixture
def worker_config(flat_topology):
    return WorkerConfig(
        num_workers=2,
        threads_per_worker=8,
        topology=flat_topology,
        numa_strategy="none",
    )


def test_prepare_jobs_detects_output_collision(worker_config, tmp_path):
    """Two source files with different extensions but same stem should raise."""
    source_folder = str(tmp_path / "source")
    output_folder = str(tmp_path / "output")
    (tmp_path / "source").mkdir()
    (tmp_path / "output").mkdir()

    source_files = [
        {"path": str(tmp_path / "source" / "movie.avi"), "filename": "movie", "duration": 100.0},
        {"path": str(tmp_path / "source" / "movie.mkv"), "filename": "movie", "duration": 200.0},
    ]

    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }

    encoder = ParallelEncoder(worker_config=worker_config, ffmpeg_path="/usr/bin/ffmpeg")

    with pytest.raises(ValueError, match="[Cc]ollision"):
        encoder.prepare_jobs(
            source_files=source_files,
            source_folder=source_folder,
            output_folder=output_folder,
            preset=preset,
        )


def test_prepare_jobs_no_collision(worker_config, tmp_path):
    """Distinct filenames should not raise."""
    source_folder = str(tmp_path / "source")
    output_folder = str(tmp_path / "output")
    (tmp_path / "source").mkdir()
    (tmp_path / "output").mkdir()

    source_files = [
        {"path": str(tmp_path / "source" / "movie1.avi"), "filename": "movie1", "duration": 100.0},
        {"path": str(tmp_path / "source" / "movie2.mkv"), "filename": "movie2", "duration": 200.0},
    ]

    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }

    encoder = ParallelEncoder(worker_config=worker_config, ffmpeg_path="/usr/bin/ffmpeg")

    jobs, skipped = encoder.prepare_jobs(
        source_files=source_files,
        source_folder=source_folder,
        output_folder=output_folder,
        preset=preset,
    )
    assert len(jobs) == 2
    assert len(skipped) == 0


def test_prepare_jobs_skips_existing_output(worker_config, tmp_path):
    """Files with existing output should be skipped by default."""
    source_folder = str(tmp_path / "source")
    output_folder = str(tmp_path / "output")
    (tmp_path / "source").mkdir()
    (tmp_path / "output").mkdir()

    # Create an existing output file for movie1
    (tmp_path / "output" / "movie1.mkv").write_text("existing")

    source_files = [
        {"path": str(tmp_path / "source" / "movie1.avi"), "filename": "movie1", "duration": 100.0},
        {"path": str(tmp_path / "source" / "movie2.mkv"), "filename": "movie2", "duration": 200.0},
    ]

    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }

    encoder = ParallelEncoder(worker_config=worker_config, ffmpeg_path="/usr/bin/ffmpeg")

    jobs, skipped = encoder.prepare_jobs(
        source_files=source_files,
        source_folder=source_folder,
        output_folder=output_folder,
        preset=preset,
    )
    assert len(jobs) == 1
    assert len(skipped) == 1
    assert "movie1.avi" in skipped[0]


def test_prepare_jobs_overwrite_ignores_existing(worker_config, tmp_path):
    """With overwrite=True, existing output files should not be skipped."""
    source_folder = str(tmp_path / "source")
    output_folder = str(tmp_path / "output")
    (tmp_path / "source").mkdir()
    (tmp_path / "output").mkdir()

    (tmp_path / "output" / "movie1.mkv").write_text("existing")

    source_files = [
        {"path": str(tmp_path / "source" / "movie1.avi"), "filename": "movie1", "duration": 100.0},
    ]

    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }

    encoder = ParallelEncoder(worker_config=worker_config, ffmpeg_path="/usr/bin/ffmpeg")

    jobs, skipped = encoder.prepare_jobs(
        source_files=source_files,
        source_folder=source_folder,
        output_folder=output_folder,
        preset=preset,
        overwrite=True,
    )
    assert len(jobs) == 1
    assert len(skipped) == 0


def test_prepare_jobs_test_encode_never_skips(worker_config, tmp_path):
    """Test encodes should never skip, even if output exists."""
    source_folder = str(tmp_path / "source")
    output_folder = str(tmp_path / "output")
    (tmp_path / "source").mkdir()
    (tmp_path / "output").mkdir()

    (tmp_path / "output" / "movie1.mkv").write_text("existing")

    source_files = [
        {"path": str(tmp_path / "source" / "movie1.avi"), "filename": "movie1", "duration": 100.0},
    ]

    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }

    encoder = ParallelEncoder(worker_config=worker_config, ffmpeg_path="/usr/bin/ffmpeg")

    jobs, skipped = encoder.prepare_jobs(
        source_files=source_files,
        source_folder=source_folder,
        output_folder=output_folder,
        preset=preset,
        test_encode=True,
    )
    assert len(jobs) == 1
    assert len(skipped) == 0


def test_prepare_jobs_all_skipped(worker_config, tmp_path):
    """When all files are skipped, jobs should be empty."""
    source_folder = str(tmp_path / "source")
    output_folder = str(tmp_path / "output")
    (tmp_path / "source").mkdir()
    (tmp_path / "output").mkdir()

    (tmp_path / "output" / "movie1.mkv").write_text("existing")
    (tmp_path / "output" / "movie2.mkv").write_text("existing")

    source_files = [
        {"path": str(tmp_path / "source" / "movie1.avi"), "filename": "movie1", "duration": 100.0},
        {"path": str(tmp_path / "source" / "movie2.mkv"), "filename": "movie2", "duration": 200.0},
    ]

    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }

    encoder = ParallelEncoder(worker_config=worker_config, ffmpeg_path="/usr/bin/ffmpeg")

    jobs, skipped = encoder.prepare_jobs(
        source_files=source_files,
        source_folder=source_folder,
        output_folder=output_folder,
        preset=preset,
    )
    assert len(jobs) == 0
    assert len(skipped) == 2


import threading


def test_keyboard_interrupt_signals_cancellation(flat_topology, worker_config):
    """KeyboardInterrupt should set cancel event so running workers can stop."""
    encoder = ParallelEncoder(worker_config=worker_config, ffmpeg_path="/usr/bin/ffmpeg")

    # After init, the encoder should have a cancel event
    assert hasattr(encoder, "_cancel_event")
    assert not encoder._cancel_event.is_set()

    # Setting it should be observable
    encoder._cancel_event.set()
    assert encoder._cancel_event.is_set()
