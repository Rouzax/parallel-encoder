"""Typed data models for parallel-encoder.

These replace the raw dicts flowing between modules, giving type safety
and clear contracts for required vs optional fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioStreamInfo:
    codec: str
    language: str
    channels: str


@dataclass(frozen=True)
class SourceFileInfo:
    path: str
    filename: str
    file_size: int
    duration: float
    video_codec: str | None
    video_width: int | None
    video_height: int | None
    video_bitrate: int | None
    video_colour_primaries: str | None
    total_bitrate: int | None
    audio_streams: list[AudioStreamInfo] = field(default_factory=list)

    @classmethod
    def from_probe_dict(cls, d: dict) -> SourceFileInfo:
        streams = [
            AudioStreamInfo(
                codec=s.get("codec", "und"),
                language=s.get("language", "und"),
                channels=s.get("channels", "und"),
            )
            for s in d.get("audio_streams", [])
        ]
        return cls(
            path=d["path"],
            filename=d["filename"],
            file_size=d.get("file_size", 0),
            duration=d.get("duration", 0.0),
            video_codec=d.get("video_codec"),
            video_width=d.get("video_width"),
            video_height=d.get("video_height"),
            video_bitrate=d.get("video_bitrate"),
            video_colour_primaries=d.get("video_colour_primaries"),
            total_bitrate=d.get("total_bitrate"),
            audio_streams=streams,
        )


@dataclass(frozen=True)
class VideoConfig:
    codec: str
    crf: int
    preset: str | int | None = None
    speed: int | None = None
    profile: str | None = None
    max_width: int | None = None
    max_height: int | None = None
    pix_fmt: str | None = None


@dataclass(frozen=True)
class AudioConfig:
    mode: str  # "passthrough" or "transcode"
    codec: str | None = None
    bitrate: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class PresetConfig:
    key: str
    display_name: str
    container: str
    video: VideoConfig
    audio: AudioConfig
    subtitles: str = "none"

    @classmethod
    def from_dict(cls, key: str, d: dict) -> PresetConfig:
        v = d["video"]
        a = d["audio"]
        return cls(
            key=key,
            display_name=d["display_name"],
            container=d.get("container", "mkv"),
            video=VideoConfig(
                codec=v["codec"],
                crf=v["crf"],
                preset=v.get("preset"),
                speed=v.get("speed"),
                profile=v.get("profile"),
                max_width=v.get("max_width"),
                max_height=v.get("max_height"),
                pix_fmt=v.get("pix_fmt"),
            ),
            audio=AudioConfig(
                mode=a["mode"],
                codec=a.get("codec"),
                bitrate=a.get("bitrate"),
                language=a.get("language"),
            ),
            subtitles=d.get("subtitles", "none"),
        )
