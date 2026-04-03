"""Tests for media_info module."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from encoder.media_info import probe_file, format_bitrate, format_size


def test_probe_file_timeout_raises():
    """ffprobe hanging should raise RuntimeError, not block forever."""
    with patch("encoder.media_info.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)
        with pytest.raises(RuntimeError, match="timed out"):
            probe_file("/fake/video.mkv")


def test_probe_file_nonzero_exit_raises():
    """ffprobe returning non-zero should raise RuntimeError."""
    with patch("encoder.media_info.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"], returncode=1, stdout="", stderr="some error"
        )
        with pytest.raises(RuntimeError, match="ffprobe failed"):
            probe_file("/fake/video.mkv")


def test_format_bitrate_megabits():
    assert format_bitrate(5_000_000) == "5.00 Mb/s"


def test_format_bitrate_none():
    assert format_bitrate(None) == "N/A"


def test_format_size_gigabytes():
    assert format_size(2_500_000_000) == "2.50 GB"


def test_format_size_zero():
    assert format_size(0) == "0 B"


from unittest.mock import patch as mock_patch


def test_probe_folder_skips_symlinks(tmp_path):
    """Symlinks should be filtered out before probing."""
    from encoder.media_info import probe_folder

    src = tmp_path / "source"
    src.mkdir()

    # Create a symlink to a file outside the source
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.mkv"
    secret.write_bytes(b"\x00" * 100)

    link = src / "sneaky.mkv"
    link.symlink_to(secret)

    # Patch probe_file so we can track what gets called
    probed_paths = []

    def tracking_probe(path):
        probed_paths.append(str(path))
        raise RuntimeError("not a real video")

    with mock_patch("encoder.media_info.probe_file", side_effect=tracking_probe):
        try:
            probe_folder(str(src), extensions=("mkv",))
        except RuntimeError:
            pass

    assert not any("sneaky" in p for p in probed_paths), \
        "Symlinked file was passed to probe_file but should have been filtered"
