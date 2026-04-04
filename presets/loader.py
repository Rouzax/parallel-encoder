"""Preset loader and FFmpeg argument builder for parallel-encoder.

Loads YAML preset definitions and converts them into FFmpeg CLI argument lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REQUIRED_VIDEO_KEYS = {"codec", "crf"}
_VALID_AUDIO_MODES = {"passthrough", "transcode"}


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
    cover_art_count: int = source_info.get("cover_art_count", 0)

    # Video streams — map all for MKV (to preserve cover art), first-only for WebM/MP4
    if container in ("mkv", "matroska") and cover_art_count > 0:
        args.extend(["-map", "0:v"])
    else:
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
    # When mapping all video streams (MKV with cover art), target only the main video
    if container in ("mkv", "matroska") and cover_art_count > 0:
        args.extend(["-c:v:0", codec])
        # Copy cover art video streams as-is and preserve attached_pic disposition
        for i in range(1, cover_art_count + 1):
            args.extend([f"-c:v:{i}", "copy", f"-disposition:v:{i}", "attached_pic"])
    else:
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

    colorspace: str | None = video.get("colorspace")
    if colorspace == "bt709":
        vf_filters.append("colorspace=all=bt709:iall=bt601-6-625")

    if vf_filters:
        # Scope filter to main video only when cover art streams are mapped
        maps_cover_art = container in ("mkv", "matroska") and cover_art_count > 0
        vf_flag = "-filter:v:0" if maps_cover_art else "-vf"
        args.extend([vf_flag, ",".join(vf_filters)])

    # ── Frame rate mode ────────────────────────────────────────
    fps_mode: str | None = video.get("fps_mode")
    if fps_mode:
        args.extend(["-fps_mode", fps_mode])

    # ── Audio codec ─────────────────────────────────────────────
    if audio["mode"] == "passthrough":
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

    return args
