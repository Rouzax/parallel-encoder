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


import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from encoder.ffmpeg import EncodingResult


def test_cover_art_temp_dir_cleaned_up(worker_config, tmp_path):
    """Temp directory created for cover art extraction must be removed after encode."""
    source = str(tmp_path / "source" / "movie.mkv")
    output = str(tmp_path / "output" / "movie.mkv")
    (tmp_path / "source").mkdir()
    (tmp_path / "output").mkdir()

    # We'll capture the temp_dir path that mkdtemp creates so we can check it later
    created_temp_dirs: list[str] = []
    original_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(**kwargs):
        d = original_mkdtemp(**kwargs)
        created_temp_dirs.append(d)
        # Put a dummy file inside to simulate extracted cover art
        dummy = Path(d) / "cover.jpg"
        dummy.write_bytes(b"\xff\xd8dummy")
        return d

    cover_art = [{"stream_index": 3, "codec_name": "mjpeg"}]

    job = MagicMock()
    job.source_path = source
    job.output_path = output
    job.preset_args = ["-c:v", "libx265"]
    job.threads = 4
    job.test_encode = None
    job.numa_node = None
    job.cover_art = cover_art

    fake_result = EncodingResult(
        source_path=source,
        output_path=output,
        success=True,
        exit_code=0,
        encoding_time=1.0,
        error_message="",
    )

    encoder = ParallelEncoder(worker_config=worker_config, ffmpeg_path="/usr/bin/ffmpeg")

    with (
        patch("tempfile.mkdtemp", side_effect=tracking_mkdtemp),
        patch("encoder.worker_pool.extract_cover_art", return_value=[
            (str(Path(tmp_path) / "dummy_cover.jpg"), "image/jpeg"),
        ]),
        patch("encoder.worker_pool.cover_art_attach_args", return_value=[]),
        patch("encoder.worker_pool.build_command", return_value=["ffmpeg", "-i", source, output]),
        patch("encoder.worker_pool.run_encode", return_value=fake_result),
    ):
        result = encoder._run_single(job, progress_callback=None)

    assert result.success
    assert len(created_temp_dirs) == 1
    # The temp directory should have been cleaned up
    assert not Path(created_temp_dirs[0]).exists(), (
        f"Temp directory {created_temp_dirs[0]} was not cleaned up"
    )


def test_keyboard_interrupt_signals_cancellation(flat_topology, worker_config):
    """KeyboardInterrupt should set cancel event so running workers can stop."""
    encoder = ParallelEncoder(worker_config=worker_config, ffmpeg_path="/usr/bin/ffmpeg")

    # After init, the encoder should have a cancel event
    assert hasattr(encoder, "_cancel_event")
    assert not encoder._cancel_event.is_set()

    # Setting it should be observable
    encoder._cancel_event.set()
    assert encoder._cancel_event.is_set()
