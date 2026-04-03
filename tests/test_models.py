"""Tests for typed data models."""

from __future__ import annotations

import pytest

from encoder.models import SourceFileInfo, PresetConfig, VideoConfig, AudioConfig


def test_source_file_info_from_probe_dict(sample_source_info):
    info = SourceFileInfo.from_probe_dict(sample_source_info)
    assert info.video_width == 1920
    assert info.video_height == 1080
    assert info.video_codec == "h264"
    assert info.duration == 3600.0
    assert len(info.audio_streams) == 1


def test_source_file_info_handles_none_dimensions():
    raw = {
        "path": "/fake/video.mkv",
        "filename": "video",
        "file_size": 100,
        "duration": 10.0,
        "video_codec": "h264",
        "video_width": None,
        "video_height": None,
        "video_bitrate": None,
        "video_colour_primaries": None,
        "total_bitrate": None,
        "audio_streams": [],
    }
    info = SourceFileInfo.from_probe_dict(raw)
    assert info.video_width is None
    assert info.video_height is None


def test_preset_config_from_dict(sample_preset):
    preset = PresetConfig.from_dict("test-key", sample_preset)
    assert preset.display_name == "Test Preset"
    assert preset.video.codec == "libx265"
    assert preset.audio.mode == "passthrough"
    assert preset.subtitles == "all"


def test_preset_config_missing_video_key():
    with pytest.raises(KeyError):
        PresetConfig.from_dict("bad", {"display_name": "X", "audio": {}})
