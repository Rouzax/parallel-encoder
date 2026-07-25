"""Preset loader and FFmpeg argument builder for parallel-encoder.

Loads YAML preset definitions and converts them into FFmpeg CLI argument lists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from presets.languages import languages_match

_log = logging.getLogger("parallel-encoder")

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

# Transfer characteristics that mean the source is HDR. These need real tone
# mapping: the `colorspace` filter only swaps matrix/primaries and cannot
# convert a PQ or HLG curve to SDR gamma.
_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})

# SDR primaries we can convert to BT.709 with the cheap `colorspace` filter,
# mapped to that filter's input-colourspace name. Anything not listed here is
# left untouched rather than guessed at.
_SDR_IALL_BY_PRIMARIES: dict[str, str] = {
    "bt470bg": "bt601-6-625",     # PAL
    "smpte170m": "bt601-6-525",   # NTSC
    "bt470m": "bt601-6-525",      # NTSC (older)
}

# Reference white for the linear-light step, in cd/m². 100 nits is the SDR
# diffuse-white convention, so anything above it becomes highlight detail for
# the tone mapper to roll off.
_TONEMAP_NOMINAL_PEAK_NITS = 100

# Tone-mapping operator. hable preserves shadow and highlight detail better
# than `reinhard` and does not clip like `clip`/`linear`. desat=0 disables the
# filter's highlight desaturation, which otherwise greys out bright colour.
_TONEMAP_OPERATOR = "hable"
_TONEMAP_DESATURATION = 0

# DV base-layer compatibility id meaning "no standalone base layer". Profile 5
# uses this: its base layer is IPT-PQ-C2, not HDR10.
_DV_BL_COMPAT_NONE = 0

# Frame side data that describes the source's HDR volume. zscale and tonemap
# convert the pixels but leave these attached, so without an explicit delete
# FFmpeg hands them to the encoder and the Matroska muxer writes them as
# colour elements. The result is a BT.709 SDR picture still advertising a
# 1000-nit BT.2020 mastering display, which can push a player into HDR mode.
_HDR_FRAME_SIDE_DATA_TO_DROP = (
    "MASTERING_DISPLAY_METADATA",
    "CONTENT_LIGHT_LEVEL",
    "DYNAMIC_HDR_PLUS",
)


class AudioLanguageNotFoundError(ValueError):
    """The source has no audio stream in the language the preset asks for.

    Raised instead of letting FFmpeg fail mid-encode, so the caller can skip
    the file up front and report it.
    """


class UnsupportedSourceColourError(ValueError):
    """The source's colour encoding cannot be converted correctly.

    Raised instead of emitting a command that would produce visibly wrong
    output, so the caller can skip the file up front and report it.
    """


def _build_colour_filters(
    video: dict,
    source_info: dict,
) -> list[str]:
    """Return the filters needed to reach the preset's target colour space.

    Only `colorspace: bt709` is handled; presets without it keep whatever the
    source uses. HDR sources get a real tone map, SDR sources with known
    non-BT.709 primaries get the cheap `colorspace` filter, and anything else
    is left alone.

    Raises UnsupportedSourceColourError for Dolby Vision profile 5, whose base
    layer decodes to a green/purple mess without the RPU applied.
    """
    if video.get("colorspace") != "bt709":
        return []

    primaries: str | None = source_info.get("video_colour_primaries")
    transfer: str | None = source_info.get("video_colour_transfer")

    if transfer in _HDR_TRANSFERS:
        # FFmpeg decodes the DV base layer and ignores the RPU. That is fine
        # for profiles 7/8/10, whose base layer is standalone HDR10, SDR or
        # HLG. Profile 5 has no standalone base layer, so the decoded pixels
        # are IPT-PQ-C2 and tone mapping them yields fluorescent green and
        # purple. Refuse instead.
        dv_profile: int | None = source_info.get("dv_profile")
        dv_compat: int | None = source_info.get("dv_bl_compatibility_id")
        if dv_profile == 5 or (dv_profile is not None and dv_compat == _DV_BL_COMPAT_NONE):
            raise UnsupportedSourceColourError(
                f"Dolby Vision profile {dv_profile} has no HDR10-compatible base "
                "layer, so it cannot be tone mapped without applying the RPU "
                "(needs dovi_tool or a libplacebo build with DV support)"
            )

        _log.debug(
            "HDR source: transfer=%s primaries=%s dv_profile=%s -> tone mapping to BT.709 SDR",
            transfer, primaries, dv_profile,
        )
        # Linearise, compress the dynamic range, then retag as BT.709 SDR.
        # tin= is set explicitly so the chain does not depend on the decoder
        # propagating the transfer tag. format=gbrpf32le is required because
        # `tonemap` only operates on linear floating-point input.
        return [
            f"zscale=tin={transfer}:t=linear:npl={_TONEMAP_NOMINAL_PEAK_NITS}",
            "format=gbrpf32le",
            f"tonemap=tonemap={_TONEMAP_OPERATOR}:desat={_TONEMAP_DESATURATION}",
            "zscale=p=bt709:t=bt709:m=bt709:r=tv",
            # Drop the now-meaningless HDR volume metadata so the output does
            # not advertise an HDR mastering display it no longer has.
            *(
                f"sidedata=mode=delete:type={sd}"
                for sd in _HDR_FRAME_SIDE_DATA_TO_DROP
            ),
        ]

    if primaries in ("bt709", None):
        # Already BT.709, or untagged and therefore not safe to reinterpret.
        return []

    iall: str | None = _SDR_IALL_BY_PRIMARIES.get(primaries)
    if iall is None:
        # Guessing here is what made HDR sources come out washed out: an
        # unrecognised value was treated as PAL and the real transfer curve
        # was never converted. Leave the source alone and say so.
        _log.warning(
            "Unrecognised colour primaries %r (transfer=%s); leaving colour "
            "untouched rather than guessing a conversion",
            primaries, transfer,
        )
        return []

    _log.debug(
        "SDR source: primaries=%s -> colorspace conversion to BT.709 (iall=%s)",
        primaries, iall,
    )
    return [f"colorspace=all=bt709:iall={iall}"]


def _select_audio_stream(audio_streams: list[dict], language: str) -> int:
    """Find the audio stream index for a requested language.

    Matching is ISO 639 aware: a preset asking for ``nld`` matches a stream
    tagged ``dut``, and vice versa (see :mod:`presets.languages`). When several
    streams match, the first is used.

    Args:
        audio_streams: ``source_info["audio_streams"]``, in source order.
        language: The language code requested by the preset.

    Returns:
        The index of the matching stream, relative to the audio streams
        (i.e. the ``N`` in FFmpeg's ``0:a:N``).

    Raises:
        AudioLanguageNotFoundError: If no audio stream matches.
    """
    for index, stream in enumerate(audio_streams):
        stream_language = stream.get("language")
        if languages_match(stream_language, language):
            _log.debug(
                "audio_language matched: requested=%s stream=0:a:%d tag=%s",
                language, index, stream_language,
            )
            return index

    available = [s.get("language") or "und" for s in audio_streams] or ["<no audio>"]
    _log.warning(
        "audio_language no match: requested=%s available=%s",
        language, ",".join(available),
    )
    raise AudioLanguageNotFoundError(
        f"preset requests {language!r} audio, but the source only has "
        f"[{', '.join(available)}]"
    )


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

    # Audio stream mapping.
    #
    # We resolve the requested language to a concrete audio-stream index rather
    # than emitting FFmpeg's "0:a:m:language:X" specifier. FFmpeg compares the
    # language tag literally, so "m:language:nld" matches nothing on a track
    # tagged "dut" (both are Dutch — 639-2/T vs 639-2/B) and aborts the encode.
    language: str | None = audio.get("language")
    if language:
        index = _select_audio_stream(source_info.get("audio_streams", []), language)
        args.extend(["-map", f"0:a:{index}"])
    else:
        args.extend(["-map", "0:a"])

    # Subtitle stream mapping
    if subtitles == "all":
        args.extend(["-map", "0:s?"])
    elif subtitles == "first":
        args.extend(["-map", "0:s:0?"])
    # "none" — no subtitle mapping

    # Attachment streams (cover art, fonts) — MKV only.
    #
    # WebM is deliberately excluded. Attachments (and attached-picture cover
    # art) are non-standard in WebM, add extra streams that media servers list
    # as additional video/attachment tracks, and serve no purpose here because
    # external sidecar artwork (-poster/-thumb/-fanart .jpg) already covers it.
    # Keeping WebM output to just video + audio (+ subtitles) avoids that.
    if container in ("mkv", "matroska"):
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

    # Build SVT-AV1 params as a single colon-separated string. FFmpeg
    # only honors the last -svtav1-params flag, so we MUST collect all
    # entries here and emit one flag. build_command will then append
    # lp=N to this single value.
    if codec == "libsvtav1":
        svt_kv: dict[str, str] = {}

        # Keyframe interval (seconds). Tightening keyint reduces seek
        # granularity and the audio scan window after seek landing in
        # WebM.
        keyint: int | None = video.get("keyint")
        if keyint is not None:
            svt_kv["keyint"] = f"{keyint}s"

        # Cap SVT-AV1 hierarchical levels for WebM output. SVT-AV1
        # >= 2.x defaults to hierarchical-levels=5, which produces a
        # long mini-GOP with frames whose DTS goes backwards on each
        # new keyframe. The matroska muxer cannot reorder these into
        # monotonic clusters, producing out-of-order WebM clusters
        # that break audio seeking in every player.
        # See SVT-AV1 issue #2351 (open) and historical fix !972
        # for #738. Capping at 3 keeps the bug from manifesting.
        if container == "webm":
            svt_kv["hierarchical-levels"] = "3"

        for k, v in video.get("svtav1_params", {}).items():
            svt_kv[str(k)] = str(v)

        if svt_kv:
            svt_str = ":".join(f"{k}={v}" for k, v in svt_kv.items())
            args.extend(["-svtav1-params", svt_str])

    # ── Video filters (scale + colorspace, combined into one -vf) ──
    vf_filters: list[str] = []

    max_w: int | None = video.get("max_width")
    max_h: int | None = video.get("max_height")
    if max_w is not None and max_h is not None:
        src_w = source_info.get("video_width")
        src_h = source_info.get("video_height")
        if src_w is not None and src_h is not None and (src_w > max_w or src_h > max_h):
            # force_divisible_by=2 rounds the auto-computed side to an even
            # number. Without it, a non-16:9 source scaled into a 16:9 box
            # (e.g. 1920x872 -> 1280x581) yields an odd dimension, which
            # x265/x264 reject under 4:2:0 chroma subsampling ("Picture
            # height must be an integer multiple of the specified chroma
            # subsampling").
            vf_filters.append(
                f"scale={max_w}:{max_h}"
                ":force_original_aspect_ratio=decrease:force_divisible_by=2"
            )

    max_fps: int | None = video.get("max_fps")
    if max_fps is not None:
        source_fps: float | None = source_info.get("video_fps")
        if source_fps is not None and source_fps > max_fps:
            vf_filters.append(f"fps={max_fps}")

    # Colour conversion runs after scaling so the expensive floating-point
    # tone-mapping work happens on the smaller frame.
    vf_filters.extend(_build_colour_filters(video, source_info))

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
    # MKV only, mirroring the attachment stream mapping above. WebM does not
    # carry attachments, so there is nothing to copy.
    if container in ("mkv", "matroska"):
        args.extend(["-c:t", "copy"])

    # ── Strip stale per-stream stats tags ─────────────────────
    # Source MKV files from mkvmerge have per-track BPS, DURATION,
    # NUMBER_OF_FRAMES, NUMBER_OF_BYTES, and _STATISTICS_* tags that
    # describe the source stream. FFmpeg copies these through to the
    # output as-is, but they no longer match the re-encoded stream.
    # MediaInfo trusts the BPS tag and reports nonsense bitrates
    # (e.g. video stream size > total file size).
    # Clearing them lets MediaInfo compute from actual stream data.
    # Setting a metadata key to empty string removes it in FFmpeg.
    if container in ("mkv", "matroska", "webm"):
        stale_tags = [
            "BPS", "DURATION",
            "NUMBER_OF_FRAMES", "NUMBER_OF_BYTES",
            "_STATISTICS_WRITING_APP",
            "_STATISTICS_WRITING_DATE_UTC",
            "_STATISTICS_TAGS",
        ]
        for stream_spec in ("v", "a", "t"):
            for tag in stale_tags:
                args.extend([f"-metadata:s:{stream_spec}", f"{tag}="])

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

    # ── MP4 fast start ────────────────────────────────────────
    # Move the moov atom to the front of the file so playback can
    # start without seeking to the end first.  This is critical for
    # streaming over SMB/HTTP where backwards seeks are slow or
    # unsupported, and is a no-cost improvement for local playback.
    if container == "mp4":
        args.extend(["-movflags", "+faststart"])

    return args
