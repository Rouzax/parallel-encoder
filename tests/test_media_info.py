"""Tests for media_info module."""

from __future__ import annotations

import json
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
            args=["ffprobe"], returncode=1, stdout=b"", stderr=b"some error"
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


def test_probe_folder_continues_on_single_file_error(tmp_path):
    """A corrupt file should not prevent other files from being probed."""
    from encoder.media_info import probe_folder
    from pathlib import Path

    src = tmp_path / "source"
    src.mkdir()
    (src / "bad.mkv").write_bytes(b"\x00" * 100)
    (src / "good.mkv").write_bytes(b"\x00" * 100)

    call_count = {"n": 0}

    def mock_probe(path, **kwargs):
        call_count["n"] += 1
        name = Path(path).name
        if "bad" in name:
            raise RuntimeError("corrupt file")
        return {"path": str(path), "filename": Path(path).stem, "duration": 10.0,
                "file_size": 100, "video_codec": "h264", "video_width": 1920,
                "video_height": 1080, "video_bitrate": None, "video_colour_primaries": None,
                "total_bitrate": None, "audio_streams": [], "cover_art_count": 0, "cover_art": []}

    with mock_patch("encoder.media_info.probe_file", side_effect=mock_probe):
        results = probe_folder(str(src), extensions=("mkv",))

    assert call_count["n"] == 2, "Both files should have been attempted"
    assert len(results) == 1, "Only the good file should be in results"
    assert results[0]["filename"] == "good"


def _probe_with_streams(streams: list[dict]) -> dict:
    """Run probe_file against a canned ffprobe JSON payload."""
    payload = json.dumps({"format": {"duration": "10.0", "size": "1000"},
                          "streams": streams}).encode()
    with patch("encoder.media_info.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout=payload, stderr=b""
        )
        return probe_file("/fake/video.mkv")


def test_probe_file_extracts_hdr_colour_properties():
    """Transfer and matrix must be probed: primaries alone cannot identify HDR."""
    info = _probe_with_streams([{
        "codec_type": "video", "codec_name": "hevc",
        "width": 3840, "height": 2160,
        "color_primaries": "bt2020",
        "color_transfer": "smpte2084",
        "color_space": "bt2020nc",
    }])
    assert info["video_colour_primaries"] == "bt2020"
    assert info["video_colour_transfer"] == "smpte2084"
    assert info["video_colour_matrix"] == "bt2020nc"


def test_probe_file_colour_properties_absent_are_none():
    """Untagged sources must report None, not a guessed value."""
    info = _probe_with_streams([{
        "codec_type": "video", "codec_name": "h264",
        "width": 1920, "height": 1080,
    }])
    assert info["video_colour_transfer"] is None
    assert info["video_colour_matrix"] is None
    assert info["dv_profile"] is None


def test_probe_file_extracts_dolby_vision_profile():
    """DV profile and base-layer compatibility drive whether we can tone map."""
    info = _probe_with_streams([{
        "codec_type": "video", "codec_name": "hevc",
        "width": 3840, "height": 2160,
        "color_primaries": "bt2020", "color_transfer": "smpte2084",
        "side_data_list": [{
            "side_data_type": "DOVI configuration record",
            "dv_profile": 8,
            "dv_bl_signal_compatibility_id": 1,
        }],
    }])
    assert info["dv_profile"] == 8
    assert info["dv_bl_compatibility_id"] == 1
