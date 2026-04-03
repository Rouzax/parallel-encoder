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
