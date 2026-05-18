# Robust VMAF Scoring for High-Resolution Content

## Problem

VMAF scoring times out on high-resolution video (3840x1920 VR180). The fixed 10-minute timeout is insufficient because VMAF must decode and compare every frame of both the source and encoded files at full resolution. A 120-second test segment at near-4K resolution exceeds the budget. There is also no progress feedback during scoring; the CLI appears frozen.

## Solution Overview

Three changes:
1. Score only a 15-second subsample from the middle of the test encode instead of the full segment.
2. Show a Rich spinner with elapsed time during scoring.
3. Scale the timeout dynamically based on resolution and sample duration, and enable multi-threaded VMAF computation.

## Detailed Design

### 1. VMAF Subsample Strategy

Add `VMAF_SAMPLE_SECONDS = 15` constant in `encoder/vmaf.py`.

The caller (`_run_vmaf_scoring()` in `encode.py`) computes a 15-second window from the middle of the test encode segment. Both the encoded file and source file are seeked to the matching window.

**Seeking math in `_run_vmaf_scoring()`:**

```
encoded_duration = dur_seconds (or full source duration if source is short)
if encoded_duration > VMAF_SAMPLE_SECONDS:
    offset = (encoded_duration - VMAF_SAMPLE_SECONDS) / 2
    vmaf_source_start = (start_seconds or 0) + offset
    vmaf_encoded_start = offset
    vmaf_duration = VMAF_SAMPLE_SECONDS
else:
    # Test encode is already <= 15s, score it all
    vmaf_source_start = start_seconds
    vmaf_encoded_start = None
    vmaf_duration = dur_seconds
```

15 seconds of representative content (sampled from the middle of the video) is sufficient for reliable comparative VMAF scoring when choosing between CRF values.

### 2. `run_vmaf()` Changes

**New parameter:** `encoded_start_seconds: float | None = None`

When provided, adds `-ss {encoded_start_seconds}` before the encoded file input in the FFmpeg command.

**`n_threads` for libvmaf:** Add `n_threads=0` to the libvmaf filter string. This tells libvmaf to auto-detect available cores and parallelize computation. Currently not set, so VMAF runs single-threaded.

Updated filter string:
```
[1:v]{scale_filter}[ref];[0:v][ref]libvmaf=log_fmt=json:log_path=-:n_threads=0
```

**Dynamic timeout:** Replace the fixed `VMAF_TIMEOUT_SECONDS = 600` with a function:

```python
def _vmaf_timeout(target_width, target_height, duration_seconds):
    """Timeout in seconds. Falls back to 120s base when duration is unknown."""
    pixel_ratio = (target_width * target_height) / (1920 * 1080)
    sample_secs = duration_seconds if duration_seconds is not None else VMAF_SAMPLE_SECONDS
    base = max(120, sample_secs * 8)
    return int(base * max(1.0, pixel_ratio))
```

Examples:
- 1920x1080, 15s: `120 * 1.0 = 120s`
- 3840x1920, 15s: `120 * 3.56 = 426s`
- 3840x2160, 15s: `120 * 4.0 = 480s`

### 3. Progress Spinner

`_run_vmaf_scoring()` in `encode.py` currently prints a static line (`"Scoring file.mp4..."`).

Change: wrap each `run_vmaf()` call with a `threading.Thread` and display a `rich.status.Status` spinner in the main thread that updates elapsed time every second.

```
Scoring file.mp4... 0:00:12
```

When VMAF completes (success or failure), the spinner stops and the result is shown in the summary table.

`run_vmaf()` itself stays synchronous with `subprocess.run()`. The threading is only at the UI layer in `encode.py`.

### 4. Dead Code Cleanup

- Remove the `VMAF_TIMEOUT_SECONDS` constant (replaced by the dynamic function).
- Remove `source_width` and `source_height` parameters from `run_vmaf()` if they are still unused after this change (currently marked `noqa: ARG001`).

## Files Changed

| File | Change |
|------|--------|
| `encoder/vmaf.py` | Add `VMAF_SAMPLE_SECONDS`, `encoded_start_seconds` param, `n_threads=0` in filter, dynamic timeout function, remove dead params |
| `encode.py` | Subsample window math in `_run_vmaf_scoring()`, spinner with elapsed time |

## Testing

- Unit tests for the dynamic timeout calculation
- Unit tests for the subsample window math (various cases: short source, exact match, long source)
- Manual test on Windows with high-res content to verify VMAF completes and spinner displays correctly
