"""Tests for preset loading and validation."""

from __future__ import annotations

import pytest

from presets.loader import (
    AudioLanguageNotFoundError,
    load_presets,
    preset_to_ffmpeg_args,
    validate_preset,
)


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


def test_preset_to_ffmpeg_args_webm_excludes_attachments():
    """WebM must not carry attachments: they are non-standard in WebM and add
    stray streams. Artwork is handled by external sidecar files instead."""
    preset = {
        "container": "webm",
        "video": {"codec": "libvpx-vp9", "crf": 30, "speed": 4},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {"video_width": 1920, "video_height": 1080, "audio_streams": []}
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "0:t?" not in args
    assert "-c:t" not in args


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


def test_preset_to_ffmpeg_args_mkv_cover_art_uses_first_video_only():
    """MKV with cover art should still map only first video (cover art re-attached post-encode)."""
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
    # Must use 0:v:0 (not 0:v) — mapping all video breaks FFmpeg progress
    assert "0:v:0" in args
    assert "-c:v" in args
    assert "-c:v:0" not in args
    # Should use -vf, not -filter:v:0
    assert "-vf" in args
    assert "-filter:v:0" not in args


def test_preset_to_ffmpeg_args_scale_forces_even_dimensions():
    """Scaling a non-16:9 source into a 16:9 box must keep dimensions even.

    A 1920x872 source fit into a 1280x720 box yields 1280x581 (odd height).
    x265 with 4:2:0 chroma subsampling rejects odd dimensions ("Picture
    height must be an integer multiple of the specified chroma subsampling").
    force_divisible_by=2 rounds the auto-computed side to an even number.
    """
    preset = {
        "container": "mkv",
        "video": {
            "codec": "libx265", "crf": 25, "preset": "faster",
            "max_width": 1280, "max_height": 720,
        },
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920, "video_height": 872,
        "audio_streams": [],
        "cover_art_count": 0,
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    vf = args[args.index("-vf") + 1]
    assert "scale=1280:720:force_original_aspect_ratio=decrease" in vf
    assert "force_divisible_by=2" in vf


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


def test_preset_to_ffmpeg_args_webm_auto_transcodes_aac_to_opus():
    """WebM with AAC source audio should auto-transcode to Opus."""
    preset = {
        "container": "webm",
        "video": {"codec": "libvpx-vp9", "crf": 30, "speed": 4},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [{"codec": "aac", "language": "eng", "channels": "2"}],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "-c:a" in args
    assert args[args.index("-c:a") + 1] == "libopus"
    assert "-b:a" in args
    assert args[args.index("-b:a") + 1] == "160k"


def test_preset_to_ffmpeg_args_webm_passthrough_opus():
    """WebM always transcodes audio to Opus (even when source is Opus) for proper seeking."""
    preset = {
        "container": "webm",
        "video": {"codec": "libvpx-vp9", "crf": 30, "speed": 4},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [{"codec": "opus", "language": "eng", "channels": "2"}],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "-c:a" in args
    assert args[args.index("-c:a") + 1] == "libopus"
    assert "-b:a" in args


def test_preset_to_ffmpeg_args_webm_opus_uses_source_bitrate():
    """WebM Opus transcode should use source bitrate when available."""
    preset = {
        "container": "webm",
        "video": {"codec": "libvpx-vp9", "crf": 30, "speed": 4},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [{"codec": "opus", "language": "eng", "channels": "2", "bit_rate": 256000}],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    assert args[args.index("-b:a") + 1] == "256k"


def test_preset_to_ffmpeg_args_mkv_passthrough_aac():
    """MKV with AAC source audio should passthrough (MKV supports AAC)."""
    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [{"codec": "aac", "language": "eng", "channels": "2"}],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "-c:a" in args
    assert args[args.index("-c:a") + 1] == "copy"


def test_preset_to_ffmpeg_args_max_fps_caps_high_framerate():
    """max_fps should add fps filter when source exceeds the cap."""
    preset = {
        "container": "webm",
        "video": {"codec": "libsvtav1", "crf": 35, "preset": 6, "max_fps": 30},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1280, "video_height": 720,
        "video_fps": 60.0,
        "audio_streams": [{"codec": "opus", "language": "eng", "channels": "2"}],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    vf_idx = args.index("-vf")
    assert "fps=30" in args[vf_idx + 1]


def test_preset_to_ffmpeg_args_max_fps_no_cap_when_below():
    """max_fps should not add fps filter when source is at or below the cap."""
    preset = {
        "container": "webm",
        "video": {"codec": "libsvtav1", "crf": 35, "preset": 6, "max_fps": 30},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1280, "video_height": 720,
        "video_fps": 24.0,
        "audio_streams": [{"codec": "opus", "language": "eng", "channels": "2"}],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    # No -vf flag at all since no scaling or fps cap needed
    full_cmd = " ".join(args)
    assert "fps=" not in full_cmd


def test_preset_to_ffmpeg_args_webm_strips_mkvmerge_stats_tags():
    """WebM/MKV output must clear stale per-stream stats tags from source.

    mkvmerge writes per-track BPS, NUMBER_OF_FRAMES, etc. that describe
    the source stream. FFmpeg copies them through, causing MediaInfo to
    report nonsense bitrates after re-encoding. We clear them.
    """
    preset = {
        "container": "webm",
        "video": {"codec": "libsvtav1", "crf": 35, "preset": 6},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1280, "video_height": 720,
        "audio_streams": [{"codec": "opus", "channels": "2"}],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    full_cmd = " ".join(args)

    # All seven stale tag keys must appear cleared on each stream type
    for stream_spec in ("v", "a", "t"):
        for tag in ("BPS", "DURATION", "NUMBER_OF_FRAMES", "NUMBER_OF_BYTES",
                    "_STATISTICS_WRITING_APP", "_STATISTICS_WRITING_DATE_UTC",
                    "_STATISTICS_TAGS"):
            assert f"-metadata:s:{stream_spec} {tag}=" in full_cmd, (
                f"missing clear for {tag} on stream {stream_spec}"
            )


def test_preset_to_ffmpeg_args_mp4_no_stats_tag_clearing():
    """MP4 output should NOT clear stats tags (they don't exist for mp4)."""
    preset = {
        "container": "mp4",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [{"codec": "aac", "channels": "2"}],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    full_cmd = " ".join(args)
    assert "-metadata:s:v BPS=" not in full_cmd
    assert "_STATISTICS_TAGS=" not in full_cmd


def test_preset_to_ffmpeg_args_webm_ignores_cover_art():
    """WebM can't hold non-VP9/AV1 video — should always map first video only and use -vf."""
    preset = {
        "container": "webm",
        "video": {
            "codec": "libvpx-vp9", "crf": 30, "speed": 4,
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
    # WebM must use 0:v:0, not 0:v
    assert "0:v:0" in args
    assert "-c:v" in args
    assert "-c:v:0" not in args
    # WebM must use -vf, not -filter:v:0 (even if source has cover art)
    assert "-vf" in args
    assert "-filter:v:0" not in args
    # WebM must not embed attachments; artwork is handled by external sidecars.
    assert "0:t?" not in args


def test_preset_to_ffmpeg_args_mkv_maps_attachments():
    """MKV keeps attachment passthrough so fonts and cover art stay embedded."""
    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium"},
        "audio": {"mode": "passthrough"},
        "subtitles": "all",
    }
    source_info = {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [],
        "cover_art_count": 1,
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "0:t?" in args


def test_preset_bt709_colorspace_skips_already_bt709():
    """Source already in BT.709 should not get colorspace filter."""
    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium", "colorspace": "bt709"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [],
        "video_colour_primaries": "bt709",
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "colorspace" not in " ".join(args), "Should skip conversion when source is already BT.709"


def test_preset_bt709_colorspace_converts_bt601():
    """Source in BT.601 should get colorspace conversion filter."""
    preset = {
        "container": "mkv",
        "video": {"codec": "libx265", "crf": 22, "preset": "medium", "colorspace": "bt709"},
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 720, "video_height": 576,
        "audio_streams": [],
        "video_colour_primaries": "bt470bg",
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    assert "-vf" in args
    vf_idx = args.index("-vf")
    assert "colorspace" in args[vf_idx + 1]
    assert "bt601-6-625" in args[vf_idx + 1]


def test_preset_to_ffmpeg_args_svtav1_params_basic():
    """svtav1_params dict should appear in -svtav1-params output."""
    preset = {
        "container": "mkv",
        "video": {
            "codec": "libsvtav1",
            "crf": 22,
            "preset": 6,
            "pix_fmt": "yuv420p10le",
            "svtav1_params": {"tune": 0},
        },
        "audio": {"mode": "passthrough"},
        "subtitles": "all",
    }
    source_info = {
        "video_width": 3840,
        "video_height": 2160,
        "audio_streams": [{"codec": "opus", "language": "eng", "channels": "2"}],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    idx = args.index("-svtav1-params")
    params_str = args[idx + 1]
    assert "tune=0" in params_str


def test_preset_to_ffmpeg_args_svtav1_params_merges_with_keyint():
    """svtav1_params should merge with computed keyint param."""
    preset = {
        "container": "mkv",
        "video": {
            "codec": "libsvtav1",
            "crf": 22,
            "preset": 6,
            "keyint": 2,
            "svtav1_params": {"tune": 0},
        },
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920,
        "video_height": 1080,
        "audio_streams": [],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    idx = args.index("-svtav1-params")
    params_str = args[idx + 1]
    assert "keyint=2s" in params_str
    assert "tune=0" in params_str


def test_preset_to_ffmpeg_args_svtav1_params_overrides_computed():
    """Explicit svtav1_params should override computed defaults like keyint."""
    preset = {
        "container": "mkv",
        "video": {
            "codec": "libsvtav1",
            "crf": 22,
            "preset": 6,
            "keyint": 2,
            "svtav1_params": {"keyint": "5s", "tune": 0},
        },
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920,
        "video_height": 1080,
        "audio_streams": [],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    idx = args.index("-svtav1-params")
    params_str = args[idx + 1]
    assert "keyint=5s" in params_str
    assert "keyint=2s" not in params_str
    assert "tune=0" in params_str


def test_preset_to_ffmpeg_args_svtav1_params_multiple():
    """Multiple svtav1_params should all appear colon-separated."""
    preset = {
        "container": "mkv",
        "video": {
            "codec": "libsvtav1",
            "crf": 22,
            "preset": 6,
            "svtav1_params": {"tune": 0, "film-grain": 8, "film-grain-denoise": 0},
        },
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1920,
        "video_height": 1080,
        "audio_streams": [],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    idx = args.index("-svtav1-params")
    params_str = args[idx + 1]
    assert "tune=0" in params_str
    assert "film-grain=8" in params_str
    assert "film-grain-denoise=0" in params_str


def test_preset_to_ffmpeg_args_svtav1_params_preserves_webm_hierarchical_levels():
    """WebM hierarchical-levels=3 cap should be preserved alongside svtav1_params."""
    preset = {
        "container": "webm",
        "video": {
            "codec": "libsvtav1",
            "crf": 32,
            "preset": 8,
            "svtav1_params": {"film-grain": 8},
        },
        "audio": {"mode": "passthrough"},
        "subtitles": "none",
    }
    source_info = {
        "video_width": 1280,
        "video_height": 720,
        "audio_streams": [{"codec": "opus", "channels": "2"}],
    }
    args = preset_to_ffmpeg_args(preset, source_info)
    idx = args.index("-svtav1-params")
    params_str = args[idx + 1]
    assert "hierarchical-levels=3" in params_str
    assert "film-grain=8" in params_str


# ---------------------------------------------------------------------------
# Tests for two-step preset selector helpers
# ---------------------------------------------------------------------------

from encode import _codec_display_name, _group_presets_by_category, _preset_short_name


def test_codec_display_name_known_codecs():
    assert _codec_display_name("libx265") == "H265 10-bit"
    assert _codec_display_name("libsvtav1") == "AV1"
    assert _codec_display_name("libx264") == "H264"
    assert _codec_display_name("libvpx-vp9") == "VP9"


def test_codec_display_name_unknown_codec():
    assert _codec_display_name("librav1e") == "librav1e"


def test_group_presets_by_category():
    presets = {
        "mkv-h265-cq20": {
            "display_name": "MKV - H265 10-bit - Medium CQ20 - All Audio Passthru",
            "container": "mkv",
            "video": {"codec": "libx265", "crf": 20, "preset": "medium"},
            "audio": {"mode": "passthrough"},
        },
        "mkv-h265-cq25": {
            "display_name": "MKV - H265 10-bit - Medium CQ25 - All Audio Passthru",
            "container": "mkv",
            "video": {"codec": "libx265", "crf": 25, "preset": "medium"},
            "audio": {"mode": "passthrough"},
        },
        "webm-av1-crf32": {
            "display_name": "WebM - 720p - AV1 - BT.709 - P8 CRF32",
            "container": "webm",
            "video": {"codec": "libsvtav1", "crf": 32, "preset": 8},
            "audio": {"mode": "passthrough"},
        },
    }
    groups = _group_presets_by_category(presets)
    assert "MKV - H265 10-bit" in groups
    assert "WebM - AV1" in groups
    assert len(groups["MKV - H265 10-bit"]) == 2
    assert len(groups["WebM - AV1"]) == 1


def test_group_presets_preserves_order():
    presets = {
        "webm-first": {
            "display_name": "WebM First",
            "container": "webm",
            "video": {"codec": "libsvtav1", "crf": 32, "preset": 8},
            "audio": {"mode": "passthrough"},
        },
        "mkv-second": {
            "display_name": "MKV Second",
            "container": "mkv",
            "video": {"codec": "libx265", "crf": 20, "preset": "medium"},
            "audio": {"mode": "passthrough"},
        },
    }
    groups = _group_presets_by_category(presets)
    categories = list(groups.keys())
    assert categories[0] == "WebM - AV1"
    assert categories[1] == "MKV - H265 10-bit"


def test_preset_short_name_strips_prefix():
    assert _preset_short_name(
        "MKV - 1080p - H265 10-bit - Medium CQ20 - All Audio Passthru",
        "MKV - H265 10-bit"
    ) == "1080p - Medium CQ20 - All Audio Passthru"


def test_preset_short_name_no_match():
    name = "WebM - 720p - AV1 - BT.709 - P8 CRF32"
    assert _preset_short_name(name, "MKV - H265 10-bit") == name


# ── Audio language selection ────────────────────────────────────────
# Regression tests for the 'nld' vs 'dut' bug: FFmpeg's m:language: matcher
# compares tags literally, so a preset asking for 'nld' failed hard on the
# entire source library, where Matroska tags every Dutch track 'dut'.

def _lang_preset(language: str) -> dict:
    return {
        "container": "mkv",
        "video": {"codec": "libsvtav1", "crf": 30, "preset": 8},
        "audio": {"mode": "passthrough", "language": language},
        "subtitles": "none",
    }


def _source(*languages: str) -> dict:
    return {
        "video_width": 1920, "video_height": 1080,
        "audio_streams": [
            {"codec": "eac3", "language": lang, "channels": "6"} for lang in languages
        ],
    }


def _audio_maps(args: list[str]) -> list[str]:
    """Extract the -map values that select audio streams."""
    return [
        args[i + 1]
        for i, a in enumerate(args)
        if a == "-map" and ":a" in args[i + 1]
    ]


def test_audio_language_nld_preset_matches_dut_tagged_stream():
    """The reported bug: preset wants 'nld', source tags Dutch as 'dut'."""
    args = preset_to_ffmpeg_args(_lang_preset("nld"), _source("eng", "dut"))
    assert _audio_maps(args) == ["0:a:1"]
    # The brittle literal-match specifier must be gone entirely.
    assert not any("m:language" in a for a in args)


def test_audio_language_selects_correct_index_when_dut_is_first():
    args = preset_to_ffmpeg_args(_lang_preset("nld"), _source("dut", "eng"))
    assert _audio_maps(args) == ["0:a:0"]


def test_audio_language_maps_only_first_match_when_several_tracks_match():
    """Two source files carry two Dutch tracks; keep only the first."""
    args = preset_to_ffmpeg_args(_lang_preset("nld"), _source("eng", "dut", "dut"))
    assert _audio_maps(args) == ["0:a:1"]


def test_audio_language_matches_regardless_of_iso_variant():
    """A 'dut' preset must equally match an 'nld'-tagged source."""
    args = preset_to_ffmpeg_args(_lang_preset("dut"), _source("eng", "nld"))
    assert _audio_maps(args) == ["0:a:1"]


def test_audio_language_english_preset_still_works():
    args = preset_to_ffmpeg_args(_lang_preset("eng"), _source("eng", "dut"))
    assert _audio_maps(args) == ["0:a:0"]


def test_audio_language_ignores_untagged_streams():
    """An untagged ('und') track must not be mistaken for the requested language."""
    with pytest.raises(AudioLanguageNotFoundError):
        preset_to_ffmpeg_args(_lang_preset("nld"), _source("und", "eng"))


def test_audio_language_no_match_raises_with_actionable_message():
    """24 library files have no Dutch track; they must fail fast, not silently."""
    with pytest.raises(AudioLanguageNotFoundError) as exc:
        preset_to_ffmpeg_args(_lang_preset("nld"), _source("eng"))
    msg = str(exc.value)
    assert "nld" in msg
    assert "eng" in msg


def test_audio_language_absent_from_preset_maps_all_audio():
    """Presets with no language filter keep the existing map-everything behaviour."""
    preset = _lang_preset("nld")
    del preset["audio"]["language"]
    args = preset_to_ffmpeg_args(preset, _source("eng", "dut"))
    assert _audio_maps(args) == ["0:a"]
