# MKV AV1 Quality Presets

## Goal

Add MKV AV1 encoding presets optimized for quality-first, then size. Target use case: 4K 60fps HDR concert recordings (YouTube VP9 sources) for playback on Vero V (Kodi, Amlogic S905X4 with full AV1 HW decode).

## New Presets

Two presets using SVT-AV1 at preset 6 with `tune=0` (SSIM/perceptual quality optimization):

### mkv-av1-p6-crf22-all-audio

Quality tier. Expected output ~9-12 GiB for a 2h42m 4K60 source (down from ~33 GiB VP9).

```yaml
mkv-av1-p6-crf22-all-audio:
  display_name: "MKV - AV1 - P6 CRF22 - All Audio Passthru"
  container: mkv
  video:
    codec: libsvtav1
    crf: 22
    preset: 6
    pix_fmt: yuv420p10le
    svtav1_params:
      tune: 0
  audio:
    mode: passthrough
  subtitles: all
```

### mkv-av1-p6-crf24-all-audio

Balanced tier. Expected output ~6-9 GiB for the same source.

```yaml
mkv-av1-p6-crf24-all-audio:
  display_name: "MKV - AV1 - P6 CRF24 - All Audio Passthru"
  container: mkv
  video:
    codec: libsvtav1
    crf: 24
    preset: 6
    pix_fmt: yuv420p10le
    svtav1_params:
      tune: 0
  audio:
    mode: passthrough
  subtitles: all
```

## Design Decisions

### HDR Preservation (no colorspace conversion)

No `colorspace` field is set. FFmpeg automatically preserves source color metadata (primaries, transfer characteristics, matrix coefficients) when no color video filters are applied. This means the preset works with HLG, HDR10, PQ, or SDR sources without modification.

### Native Resolution (no cap)

No `max_width`/`max_height` set. Consistent with the existing `mkv-h265-10bit-*` native resolution presets. The output resolution matches the source.

### tune=0 (Visual Quality)

SVT-AV1's default is `tune=1` (PSNR). `tune=0` optimizes for SSIM/perceptual quality, redistributing bits to perceptually important areas. Always beneficial for content meant for human viewing.

### No film-grain by default

The primary sources are YouTube VP9 re-encodes. Film grain synthesis would misidentify compression artifacts as grain. The `svtav1_params` mechanism makes it trivial to add `film-grain` and `film-grain-denoise` for presets targeting clean camera originals in the future.

### Audio passthrough

MKV supports all audio codecs natively. No reason to re-encode, especially for Opus sources that are already efficiently compressed.

### Subtitles: all

Consistent with existing native-resolution MKV presets.

## Code Changes

### 1. presets/loader.py - Generic svtav1_params support

Extend the SVT-AV1 parameter building block in `preset_to_ffmpeg_args()` to read an optional `video.svtav1_params` dict from the preset YAML. Dict entries become `key=value` pairs joined with colons, merged with the computed params (keyint, hierarchical-levels). YAML-specified params take precedence over computed defaults on conflict. The thread parameter `lp=N` is appended later by `build_command()`.

Current code builds svtav1_params from `keyint` and `hierarchical-levels` (for webm). The change adds a merge step:

```python
# Existing: svt_params list built from keyint, hierarchical-levels
# New: merge in svtav1_params dict from preset YAML
extra_params: dict = video.get("svtav1_params", {})
for k, v in extra_params.items():
    svt_params.append(f"{k}={v}")
```

This must be placed after the existing keyint/hierarchical-levels logic but before the final `-svtav1-params` emit, so that YAML values can override computed defaults if needed.

### 2. config/presets.yaml - Add presets

Add a new section with the two presets after the existing native-resolution H.265 presets.

## Testing

- Verify presets load without validation errors: `python -c "from presets.loader import load_presets; load_presets('config/presets.yaml')"`
- Verify `preset_to_ffmpeg_args()` emits `-svtav1-params` containing `tune=0` for the new presets
- Verify existing presets with `keyint` still work (regression check)
- Verify `build_command()` correctly appends `lp=N` to the params string that now includes `tune=0`
- Use `--dry-run` to inspect generated FFmpeg commands
- Use `--test-encode` on a short segment of a 4K HDR source to validate HDR metadata preservation in output
