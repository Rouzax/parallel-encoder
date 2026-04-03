"""Shared fixtures for parallel-encoder tests."""

from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture
def tmp_source(tmp_path: Path) -> Path:
    """Create a temporary source directory."""
    src = tmp_path / "source"
    src.mkdir()
    return src


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def sample_source_info() -> dict:
    """Return a realistic probe_file result for testing."""
    return {
        "path": "/fake/source/video.mkv",
        "filename": "video",
        "file_size": 1_000_000_000,
        "duration": 3600.0,
        "video_codec": "h264",
        "video_width": 1920,
        "video_height": 1080,
        "video_bitrate": 5_000_000,
        "video_colour_primaries": "bt709",
        "total_bitrate": 6_000_000,
        "audio_streams": [
            {"codec": "aac", "language": "eng", "channels": "2"},
        ],
    }


@pytest.fixture
def sample_preset() -> dict:
    """Return a minimal valid preset dict for testing."""
    return {
        "display_name": "Test Preset",
        "container": "mkv",
        "video": {
            "codec": "libx265",
            "profile": "main10",
            "crf": 22,
            "preset": "medium",
            "max_width": 1920,
            "max_height": 1080,
            "pix_fmt": "yuv420p10le",
        },
        "audio": {
            "mode": "passthrough",
        },
        "subtitles": "all",
    }
