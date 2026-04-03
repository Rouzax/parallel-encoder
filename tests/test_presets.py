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
