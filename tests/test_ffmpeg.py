"""Tests for ffmpeg command building and atomic output."""

from __future__ import annotations

from encoder.ffmpeg import build_command, atomic_output_path, finalize_output, _parse_progress_line, _parse_time


def test_build_command_basic():
    cmd = build_command(
        ffmpeg_path="/usr/bin/ffmpeg",
        source="/in/video.mkv",
        output="/out/video.mkv",
        preset_args=["-c:v", "libx265", "-crf", "22"],
        threads=8,
    )
    assert cmd[0] == "/usr/bin/ffmpeg"
    assert "-i" in cmd
    assert "/in/video.mkv" in cmd
    assert "/out/video.mkv" in cmd


def test_build_command_x265_pools():
    cmd = build_command(
        ffmpeg_path="ffmpeg",
        source="in.mkv",
        output="out.mkv",
        preset_args=["-c:v", "libx265", "-crf", "22"],
        threads=16,
    )
    joined = " ".join(cmd)
    assert "pools=16" in joined


def test_build_command_svtav1_lp():
    cmd = build_command(
        ffmpeg_path="ffmpeg",
        source="in.mkv",
        output="out.mkv",
        preset_args=["-c:v", "libsvtav1", "-crf", "28"],
        threads=20,
    )
    joined = " ".join(cmd)
    assert "lp=20" in joined


def test_build_command_test_encode():
    cmd = build_command(
        ffmpeg_path="ffmpeg",
        source="in.mkv",
        output="out.mkv",
        preset_args=["-c:v", "libx264"],
        threads=4,
        test_encode={"start_seconds": 60, "duration_seconds": 120},
    )
    joined = " ".join(cmd)
    assert "-ss 60" in joined
    assert "-t 120" in joined


def test_build_command_x264_no_duplicate_threads():
    """x264 should not get both -x264-params threads=N and -threads N."""
    cmd = build_command(
        ffmpeg_path="ffmpeg",
        source="in.mkv",
        output="out.mkv",
        preset_args=["-c:v", "libx264", "-crf", "22"],
        threads=8,
    )
    joined = " ".join(cmd)
    assert "threads=8" in joined
    assert joined.count("-threads") == 0, "x264 should use -x264-params threads=N only, not -threads"


def test_atomic_output_path(tmp_path):
    output = str(tmp_path / "video.mkv")
    temp = atomic_output_path(output)
    assert temp.endswith(".tmp.mkv")
    assert "video" in temp


def test_finalize_output_renames(tmp_path):
    temp_file = tmp_path / "video.tmp.mkv"
    temp_file.write_text("data")
    final = str(tmp_path / "video.mkv")
    finalize_output(str(temp_file), final)
    from pathlib import Path
    assert Path(final).exists()
    assert not temp_file.exists()


# ---------------------------------------------------------------------------
# Progress parsing tests
# ---------------------------------------------------------------------------


def test_parse_progress_line_standard():
    line = "frame=  100 fps= 25.0 q=28.0 Lsize=    5000kB time=00:00:04.00 bitrate=1234.5kbits/s speed=1.50x"
    result = _parse_progress_line(line)
    assert result is not None
    assert result["frame"] == 100
    assert result["fps"] == 25.0
    assert result["time_seconds"] == 4.0
    assert result["speed"] == 1.50


def test_parse_progress_line_size_without_L():
    """FFmpeg < 8.x uses 'size=' without 'L' prefix."""
    line = "frame=   50 fps= 10.0 q=22.0 size=    2500kB time=00:00:02.00 bitrate= 500.0kbits/s speed=0.80x"
    result = _parse_progress_line(line)
    assert result is not None
    assert result["frame"] == 50


def test_parse_progress_line_returns_none_for_non_progress():
    assert _parse_progress_line("Stream #0:0: Video: h264") is None
    assert _parse_progress_line("") is None


def test_parse_time_valid():
    assert _parse_time("01:30:45.50") == 5445.5


def test_parse_time_malformed_returns_zero():
    assert _parse_time("invalid") == 0.0
