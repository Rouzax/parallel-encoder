"""Tests for preset loading and validation."""

from __future__ import annotations

import pytest

from presets.loader import load_presets, validate_preset, preset_to_ffmpeg_args


def test_validate_preset_valid(sample_preset):
    """A well-formed preset should pass validation."""
    validate_preset("test-key", sample_preset)  # should not raise


def test_validate_preset_missing_video():
    with pytest.raises(ValueError, match="video"):
        validate_preset("bad", {"display_name": "X", "audio": {"mode": "passthrough"}})


def test_validate_preset_missing_codec():
    with pytest.raises(ValueError, match="codec"):
        validate_preset("bad", {
            "display_name": "X",
            "video": {"crf": 22, "preset": "medium"},
            "audio": {"mode": "passthrough"},
        })


def test_validate_preset_missing_audio():
    with pytest.raises(ValueError, match="audio"):
        validate_preset("bad", {
            "display_name": "X",
            "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        })


def test_validate_preset_missing_display_name():
    with pytest.raises(ValueError, match="display_name"):
        validate_preset("bad", {
            "video": {"codec": "libx265", "crf": 22},
            "audio": {"mode": "passthrough"},
        })


def test_preset_to_ffmpeg_args_with_none_dimensions():
    """When source has None dimensions and preset has max_width/height, should not crash."""
    preset = {
        "video": {
            "codec": "libx265",
            "crf": 22,
            "preset": "medium",
            "max_width": 1920,
            "max_height": 1080,
            "profile": "main10",
            "pix_fmt": "yuv420p10le",
        },
        "audio": {"mode": "passthrough"},
        "subtitles": "all",
    }
    source_info = {
        "video_width": None,
        "video_height": None,
        "audio_streams": [],
    }
    # Should not raise; should skip scaling when dimensions are unknown
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "-vf" not in args


def test_load_presets_from_yaml(tmp_path):
    yaml_file = tmp_path / "presets.yaml"
    yaml_file.write_text("""
presets:
  test:
    display_name: "Test"
    container: mkv
    video:
      codec: libx265
      crf: 22
      preset: medium
    audio:
      mode: passthrough
""")
    presets = load_presets(yaml_file)
    assert "test" in presets
    assert presets["test"]["display_name"] == "Test"


def test_preset_to_ffmpeg_args_mkv_includes_attachments():
    """MKV presets should map and copy attachment streams."""
    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {"video_width": 1920, "video_height": 1080, "audio_streams": []}
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "0:t?" in args
    assert "-c:t" in args
    assert args[args.index("-c:t") + 1] == "copy"


def test_preset_to_ffmpeg_args_webm_includes_attachments():
    """WebM uses the same Matroska muxer, so it should also include attachments."""
    preset = {
        "container": "webm",
        "video": {"codec": "libvpx-vp9", "crf": 30, "speed": 4},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {"video_width": 1920, "video_height": 1080, "audio_streams": []}
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "0:t?" in args
    assert "-c:t" in args


def test_preset_to_ffmpeg_args_mp4_excludes_attachments():
    """MP4 does not support attachments — should not include attachment mapping."""
    preset = {
        "container": "mp4",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {"video_width": 1920, "video_height": 1080, "audio_streams": []}
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "0:t?" not in args
    assert "-c:t" not in args


def test_preset_to_ffmpeg_args_default_container_includes_attachments():
    """Preset without explicit container defaults to mkv — should include attachments."""
    preset = {
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {"video_width": 1920, "video_height": 1080, "audio_streams": []}
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "0:t?" in args
    assert "-c:t" in args


def test_preset_to_ffmpeg_args_mkv_cover_art_maps_all_video():
    """MKV with cover art should map all video streams and use per-stream codecs."""
    preset = {
        "container": "mkv",
        "video": {
            "codec": "libx265", "crf": 22, "preset": "medium",
            "max_width": 1280, "max_height": 720,
        },
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [],
        "cover_art_count": 1,
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    # Should map all video (not just 0:v:0)
    assert "0:v" in args
    assert "0:v:0" not in args
    # Should use per-stream codec for main video and copy for cover art
    assert "-c:v:0" in args
    assert args[args.index("-c:v:0") + 1] == "libx265"
    assert "-c:v:1" in args
    assert args[args.index("-c:v:1") + 1] == "copy"
    # Video filters should be scoped to stream 0 only
    assert "-filter:v:0" in args
    assert "-vf" not in args


def test_preset_to_ffmpeg_args_mkv_no_cover_art_maps_first_video():
    """MKV without cover art should map only first video stream."""
    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [],
        "cover_art_count": 0,
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "0:v:0" in args
    assert "-c:v" in args
    assert "-c:v:0" not in args


def test_preset_to_ffmpeg_args_webm_ignores_cover_art():
    """WebM can't hold non-VP9/AV1 video — should always map first video only."""
    preset = {
        "container": "webm",
        "video": {"codec": "libvpx-vp9", "crf": 30, "speed": 4},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [],
        "cover_art_count": 1,
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    # WebM must use 0:v:0, not 0:v
    assert "0:v:0" in args
    assert "-c:v" in args
    assert "-c:v:0" not in args
