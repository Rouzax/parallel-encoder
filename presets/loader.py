"""Preset loader and FFmpeg argument builder for parallel-encoder.

Loads YAML preset definitions and converts them into FFmpeg CLI argument lists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger("parallel-encoder")

import yaml

# Default Opus bitrates per channel count when source bitrate is unknown.
_OPUS_BITRATE_BY_CHANNELS: dict[int, str] = {
    1: "96k",
    2: "160k",
    6: "256k",   # 5.1
    8: "384k",   # 7.1
}
_OPUS_BITRATE_DEFAULT = "160k"

_REQUIRED_VIDEO_KEYS = {"codec", "crf"}
_VALID_AUDIO_MODES = {"passthrough", "transcode"}


def _pick_opus_bitrate(audio_streams: list[dict]) -> str:
    """Choose an Opus bitrate based on the source audio streams.

    Uses the source bitrate if available (from ffprobe), otherwise
    falls back to a sensible default based on channel count.
    """
    if not audio_streams:
        return _OPUS_BITRATE_DEFAULT

    # Use the first audio stream's properties
    stream = audio_streams[0]
    source_bps: int | None = stream.get("bit_rate")

    if source_bps is not None and source_bps > 0:
        # Round to nearest 1k for a clean value
        kbps = max(64, round(source_bps / 1000))
        return f"{kbps}k"

    # Fallback: pick by channel count
    try:
        channels = int(stream.get("channels", 2))
    except (ValueError, TypeError):
        channels = 2
    return _OPUS_BITRATE_BY_CHANNELS.get(channels, _OPUS_BITRATE_DEFAULT)


def validate_preset(key: str, preset: dict) -> None:
    """Validate a preset dict and raise ValueError with a clear message on problems.

    Args:
        key: The preset key (used in error messages).
        preset: The preset configuration dict.

    Raises:
        ValueError: If the preset is missing required fields.
    """
    if "display_name" not in preset:
        raise ValueError(f"Preset '{key}': missing 'display_name'")

    if "video" not in preset:
        raise ValueError(f"Preset '{key}': missing 'video' section")

    video = preset["video"]
    for req in _REQUIRED_VIDEO_KEYS:
        if req not in video:
            raise ValueError(f"Preset '{key}': missing video.{req}")

    if "audio" not in preset:
        raise ValueError(f"Preset '{key}': missing 'audio' section")

    audio = preset["audio"]
    if "mode" not in audio:
        raise ValueError(f"Preset '{key}': missing audio.mode")

    if audio["mode"] not in _VALID_AUDIO_MODES:
        raise ValueError(
            f"Preset '{key}': audio.mode must be one of {_VALID_AUDIO_MODES}, "
            f"got '{audio['mode']}'"
        )

    if audio["mode"] == "transcode" and "codec" not in audio:
        raise ValueError(f"Preset '{key}': audio.mode is 'transcode' but no audio.codec specified")


def load_presets(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load presets from a YAML file.

    Args:
        path: Path to the presets YAML file.

    Returns:
        Dictionary mapping preset key to its configuration dict.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
        KeyError: If the top-level ``presets`` key is missing.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if "presets" not in data:
        raise KeyError("YAML file is missing the top-level 'presets' key")
    presets = data["presets"]
    for key, cfg in presets.items():
        validate_preset(key, cfg)
    return presets


def list_preset_names(presets: dict[str, dict[str, Any]]) -> list[str]:
    """Return a sorted list of human-readable display names.

    Args:
        presets: The dict returned by :func:`load_presets`.

    Returns:
        Alphabetically sorted list of ``display_name`` values.
    """
    return sorted(cfg["display_name"] for cfg in presets.values())


def get_preset_by_name(
    presets: dict[str, dict[str, Any]], display_name: str
) -> tuple[str, dict[str, Any]]:
    """Look up a preset by its display name.

    Args:
        presets: The dict returned by :func:`load_presets`.
        display_name: The human-readable name to search for.

    Returns:
        A ``(key, config)`` tuple for the matching preset.

    Raises:
        ValueError: If no preset matches the given display name.
    """
    for key, cfg in presets.items():
        if cfg["display_name"] == display_name:
            return key, cfg
    raise ValueError(f"No preset with display_name '{display_name}'")


def preset_to_ffmpeg_args(
    preset: dict[str, Any],
    source_info: dict[str, Any],
) -> list[str]:
    """Convert a preset configuration and source media info into FFmpeg CLI args.

    The returned list is suitable for passing to ``subprocess.run(["ffmpeg", ...] + args)``.
    It does **not** include the ``ffmpeg`` binary, the input ``-i`` flag, or the output path.

    Args:
        preset: A single preset configuration dict (the *value* from :func:`load_presets`).
        source_info: Information about the source file with keys:
            - ``video_width``  (int): source video width in pixels
            - ``video_height`` (int): source video height in pixels
            - ``audio_streams`` (list[dict]): each dict has ``codec``, ``language``, ``channels``

    Returns:
        List of FFmpeg CLI argument strings.
    """
    args: list[str] = []
    video: dict[str, Any] = preset["video"]
    audio: dict[str, Any] = preset["audio"]
    subtitles: str = preset.get("subtitles", "none")
    container: str = preset.get("container", "mkv").lower()

    # ── Stream mapping ──────────────────────────────────────────
    # Map first video stream only. Cover art (attached_pic video streams)
    # cannot be mapped alongside encoded video — FFmpeg outputs time=N/A
    # which breaks progress reporting. Cover art is re-attached post-encode.
    args.extend(["-map", "0:v:0"])

    # Audio stream mapping
    language: str | None = audio.get("language")
    if language:
        args.extend(["-map", f"0:a:m:language:{language}"])
    else:
        args.extend(["-map", "0:a"])

    # Subtitle stream mapping
    if subtitles == "all":
        args.extend(["-map", "0:s?"])
    elif subtitles == "first":
        args.extend(["-map", "0:s:0?"])
    # "none" — no subtitle mapping

    # Attachment streams (cover art, fonts) — Matroska-based containers only
    if container in ("mkv", "matroska", "webm"):
        args.extend(["-map", "0:t?"])

    # ── Video codec ─────────────────────────────────────────────
    codec: str = video["codec"]
    args.extend(["-c:v", codec])
    args.extend(["-crf", str(video["crf"])])

    if codec == "libvpx-vp9":
        # VP9 uses -speed instead of -preset, and needs -b:v 0 for CRF mode
        args.extend(["-b:v", "0"])
        args.extend(["-speed", str(video["speed"])])
    elif codec == "libsvtav1":
        args.extend(["-preset", str(video["preset"])])
    else:
        # libx265 / libx264
        args.extend(["-preset", str(video["preset"])])

    # Pixel format
    if "pix_fmt" in video:
        args.extend(["-pix_fmt", video["pix_fmt"]])

    # Codec-specific profile params
    if video.get("profile"):
        args.extend(["-profile:v", video["profile"]])

    # Keyframe interval (in seconds).  For SVT-AV1 this is added to
    # svtav1-params; build_command merges with the lp= entry.
    # Tightening keyint reduces seek granularity and the audio scan
    # window after seek landing in WebM.
    keyint: int | None = video.get("keyint")
    if keyint is not None and codec == "libsvtav1":
        args.extend(["-svtav1-params", f"keyint={keyint}s"])

    # ── Video filters (scale + colorspace, combined into one -vf) ──
    vf_filters: list[str] = []

    max_w: int | None = video.get("max_width")
    max_h: int | None = video.get("max_height")
    if max_w is not None and max_h is not None:
        src_w = source_info.get("video_width")
        src_h = source_info.get("video_height")
        if src_w is not None and src_h is not None and (src_w > max_w or src_h > max_h):
            vf_filters.append(
                f"scale={max_w}:{max_h}:force_original_aspect_ratio=decrease"
            )

    max_fps: int | None = video.get("max_fps")
    if max_fps is not None:
        source_fps: float | None = source_info.get("video_fps")
        if source_fps is not None and source_fps > max_fps:
            vf_filters.append(f"fps={max_fps}")

    colorspace: str | None = video.get("colorspace")
    if colorspace == "bt709":
        source_primaries = source_info.get("video_colour_primaries")
        # Skip conversion if source is already BT.709 or unknown
        if source_primaries not in ("bt709", None):
            # Map known primaries to FFmpeg's colorspace input name
            iall_map = {
                "bt470bg": "bt601-6-625",       # PAL
                "smpte170m": "bt601-6-525",      # NTSC
                "bt470m": "bt601-6-525",         # NTSC (older)
            }
            iall = iall_map.get(source_primaries, "bt601-6-625")
            vf_filters.append(f"colorspace=all=bt709:iall={iall}")

    if vf_filters:
        # Scope filter to main video only when cover art streams are mapped
        args.extend(["-vf", ",".join(vf_filters)])

    # ── Frame rate mode ────────────────────────────────────────
    fps_mode: str | None = video.get("fps_mode")
    if fps_mode:
        args.extend(["-fps_mode", fps_mode])

    # ── Audio codec ─────────────────────────────────────────────
    if audio["mode"] == "passthrough":
        if container == "webm":
            # WebM requires Opus/Vorbis.  Always transcode to Opus rather
            # than stream-copying, even when the source is already Opus.
            # Stream-copied Opus in WebM causes ~5s audio delay on seeking
            # because FFmpeg doesn't rewrite packet headers for the new
            # container's cluster boundaries.
            bitrate = _pick_opus_bitrate(source_info.get("audio_streams", []))
            _log.info("Transcoding audio to Opus @ %s for WebM", bitrate)
            args.extend(["-c:a", "libopus", "-b:a", bitrate])
        else:
            args.extend(["-c:a", "copy"])
    else:
        args.extend(["-c:a", audio["codec"]])
        if "bitrate" in audio:
            args.extend(["-b:a", audio["bitrate"]])

    # ── Subtitles codec ─────────────────────────────────────────
    if subtitles in ("all", "first"):
        args.extend(["-c:s", "copy"])

    # ── Attachments codec ──────────────────────────────────────
    if container in ("mkv", "matroska", "webm"):
        args.extend(["-c:t", "copy"])

    # ── WebM seeking optimisation ─────────────────────────────
    # In multi-track WebM, FFmpeg only writes Cue points for video
    # keyframes, never for audio. When seeking, the demuxer lands on
    # a video cue then linear-scans audio within the cluster.
    #
    # Fix:
    #   -cues_to_front 1           place the Cues element at the start
    #                              of the file for fast HTTP seek
    #   -reserve_index_space N     reserve space so cues_to_front does
    #                              not require shifting the file. Size
    #                              scales with duration.
    #
    # We rely on the encoder's keyint to control cluster size (since
    # the matroska muxer creates a new cluster on each video keyframe).
    # WebM AV1 presets set keyint=2 -> 2s clusters -> ~2s audio scan
    # window after seek.
    #
    # NOTE: We do NOT use -cluster_time_limit. SVT-AV1 with high
    # parallelism (lp=6) buffers many frames of video lookahead. If
    # the muxer flushes clusters on a fixed time interval shorter
    # than the encoder's lookahead, audio packets stream out before
    # video catches up, producing clusters with no video and later
    # clusters with backdated video timestamps. The result is an
    # out-of-order WebM that breaks audio seeking in every player.
    #
    # NOTE: -dash 1 is NOT used either. It hard-requires single-track
    # output (nb_tracks == 1) and would fail every multi-track encode
    # with EINVAL.
    if container == "webm":
        duration: float = source_info.get("duration", 0.0) or 0.0
        # ~1 cue per video keyframe, ~20 bytes per cue, 100% safety.
        # At keyint=2s the cue density is ~0.5 cues/sec; we use 1/sec
        # as a conservative upper bound that also covers shorter keyint.
        needed = int(duration * 20 * 2)
        # Round up to nearest 64 KiB, minimum 256 KiB.
        reserve = max(262144, ((needed + 65535) // 65536) * 65536)
        args.extend([
            "-cues_to_front", "1",
            "-reserve_index_space", str(reserve),
        ])

    return args
