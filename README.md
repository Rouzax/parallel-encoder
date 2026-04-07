# parallel-encoder

Parallel video encoder CLI that runs multiple FFmpeg processes concurrently. It probes source video files with ffprobe, selects an encoding preset from YAML config, auto-detects CPU/NUMA topology to determine optimal worker count and thread distribution, then encodes files in parallel with real-time progress display.

## Features

- **Parallel encoding** with automatic worker count based on CPU topology
- **NUMA-aware** thread pinning on multi-socket systems (Linux via `numactl`, Windows via processor group affinity)
- **YAML presets** for repeatable encoding profiles (AV1, x265, x264, VP9)
- **Codec-specific threading** (SVT-AV1 `lp`, x265 `pools`, x264 `threads`, VP9 `tile-columns`)
- **Smart defaults** - never upscales, auto-transcodes incompatible audio for WebM, caps frame rate when configured
- **Rich progress display** with per-file and overall progress bars, ETA, fps, and speed multiplier
- **Test encode mode** to encode a short segment before committing to a full run
- **Graceful cancellation** with a single Ctrl+C

## Requirements

- Python >= 3.10
- FFmpeg and ffprobe on PATH
- Optional: `numactl` on Linux for NUMA pinning on multi-socket systems

## Installation

```bash
pip install -e .
```

This installs the `parallel-encode` CLI entry point.

## Usage

```bash
# Basic usage
parallel-encode -s /path/to/source -o /path/to/output

# Or run directly
python encode.py -s /path/to/source -o /path/to/output

# Select a specific preset
parallel-encode -s /source -o /output -p webm-720p-av1-bt709-p6-crf35

# Override worker count
parallel-encode -s /source -o /output -w 2

# Test encode (short segment from the middle of each file)
parallel-encode -s /source -o /output --test-encode

# Dry run (print FFmpeg commands without executing)
parallel-encode -s /source -o /output --dry-run

# Verbose output
parallel-encode -s /source -o /output -vv

# Log to file
parallel-encode -s /source -o /output --log-file encode.log

# Copy non-video files (artwork, NFO, etc.) to output
parallel-encode -s /source -o /output --copy-all
```

If no preset is specified with `-p`, an interactive selector lets you pick one grouped by container and codec.

## Presets

Presets are defined in `config/presets.yaml`. Each preset specifies:

```yaml
presets:
  webm-720p-av1-bt709-p6-crf35:
    display_name: "WebM - 720p - AV1 - BT.709 - P6 CRF35"
    container: webm
    video:
      codec: libsvtav1
      crf: 35
      preset: 6
      max_width: 1280
      max_height: 720
      max_fps: 30          # optional, caps frame rate (won't upscale lower fps)
      pix_fmt: yuv420p
      colorspace: bt709
    audio:
      mode: passthrough    # or transcode with codec/bitrate
    subtitles: none        # all, first, or none
```

## NUMA Support

On multi-socket systems, the encoder automatically detects NUMA topology and pins each worker to a specific NUMA node to avoid cross-socket memory penalties.

- **Linux**: Uses `numactl --cpunodebind --membind` to wrap FFmpeg commands
- **Windows**: Uses `SetProcessAffinityMask` (same processor group) and `SetThreadGroupAffinity` (cross-group) with re-pinning after encoder thread creation

## Testing

```bash
pip install -e ".[dev]"
python -m pytest
```
