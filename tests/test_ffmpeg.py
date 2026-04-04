"""Tests for ffmpeg command building and atomic output."""

from __future__ import annotations

from encoder.ffmpeg import build_command, atomic_output_path, finalize_output


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


def test_windows_numa_affinity_mask_calculation():
    """Verify the CPU mask calculation for NUMA node pinning."""
    # Node 0, 40 threads per node: CPUs 0-39 → mask = (1<<40)-1 = 0xFFFFFFFFFF
    node, tpn = 0, 40
    mask = ((1 << tpn) - 1) << (node * tpn)
    assert mask == 0xFFFFFFFFFF

    # Node 1, 40 threads per node: CPUs 40-79 → mask shifted left by 40
    node, tpn = 1, 40
    mask = ((1 << tpn) - 1) << (node * tpn)
    assert mask == 0xFFFFFFFFFF << 40

    # Node 0, 8 threads per node: CPUs 0-7
    node, tpn = 0, 8
    mask = ((1 << tpn) - 1) << (node * tpn)
    assert mask == 0xFF

    # Node 1, 8 threads per node: CPUs 8-15
    node, tpn = 1, 8
    mask = ((1 << tpn) - 1) << (node * tpn)
    assert mask == 0xFF00
