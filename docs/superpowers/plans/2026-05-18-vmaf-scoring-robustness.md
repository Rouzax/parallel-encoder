# Robust VMAF Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VMAF scoring reliable on high-resolution content by subsampling to 15 seconds, adding a progress spinner, and scaling the timeout dynamically.

**Architecture:** `encoder/vmaf.py` gains encoded-file seeking, multi-threaded libvmaf, and a dynamic timeout function. `encode.py` computes the 15-second subsample window and wraps each VMAF call with a Rich spinner running in the main thread while VMAF executes in a background thread.

**Tech Stack:** Python 3.10+, FFmpeg libvmaf, Rich (Status spinner), threading

---

### Task 1: Dynamic timeout and n_threads in `encoder/vmaf.py`

**Files:**
- Modify: `encoder/vmaf.py:12` (replace constant), `encoder/vmaf.py:50-53` (filter string), `encoder/vmaf.py:75` (timeout call)
- Test: `tests/test_vmaf.py` (create)

- [ ] **Step 1: Write failing tests for `_vmaf_timeout`**

Create `tests/test_vmaf.py`:

```python
"""Tests for VMAF scoring helpers."""

from __future__ import annotations

from encoder.vmaf import _vmaf_timeout, VMAF_SAMPLE_SECONDS


def test_vmaf_timeout_1080p_default():
    result = _vmaf_timeout(1920, 1080, 15)
    assert result == 120


def test_vmaf_timeout_4k_scales_up():
    result = _vmaf_timeout(3840, 2160, 15)
    assert result == 480


def test_vmaf_timeout_vr180_scales_up():
    result = _vmaf_timeout(3840, 1920, 15)
    expected = int(120 * (3840 * 1920) / (1920 * 1080))
    assert result == expected


def test_vmaf_timeout_long_duration_raises_base():
    result = _vmaf_timeout(1920, 1080, 60)
    assert result == 480


def test_vmaf_timeout_none_duration_uses_sample_default():
    result = _vmaf_timeout(1920, 1080, None)
    assert result == 120


def test_vmaf_timeout_below_1080p_clamps_to_base():
    result = _vmaf_timeout(1280, 720, 15)
    assert result == 120


def test_vmaf_sample_seconds_is_15():
    assert VMAF_SAMPLE_SECONDS == 15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vmaf.py -v`
Expected: ImportError for `_vmaf_timeout` and `VMAF_SAMPLE_SECONDS`

- [ ] **Step 3: Implement `_vmaf_timeout` and `VMAF_SAMPLE_SECONDS`**

In `encoder/vmaf.py`, replace line 12:

```python
VMAF_TIMEOUT_SECONDS = 600  # 10 minutes per comparison
```

with:

```python
VMAF_SAMPLE_SECONDS = 15


def _vmaf_timeout(
    target_width: int, target_height: int, duration_seconds: float | None
) -> int:
    pixel_ratio = (target_width * target_height) / (1920 * 1080)
    sample_secs = duration_seconds if duration_seconds is not None else VMAF_SAMPLE_SECONDS
    base = max(120, int(sample_secs * 8))
    return int(base * max(1.0, pixel_ratio))
```

- [ ] **Step 4: Add `n_threads=0` to the libvmaf filter string**

In `encoder/vmaf.py`, change the `vmaf_filter` construction (lines 50-53) from:

```python
    vmaf_filter = (
        f"[1:v]{scale_filter}[ref];"
        f"[0:v][ref]libvmaf=log_fmt=json:log_path=-"
    )
```

to:

```python
    vmaf_filter = (
        f"[1:v]{scale_filter}[ref];"
        f"[0:v][ref]libvmaf=log_fmt=json:log_path=-:n_threads=0"
    )
```

- [ ] **Step 5: Update `subprocess.run` to use dynamic timeout**

In `encoder/vmaf.py`, change the `subprocess.run` call (line 72-76) from:

```python
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=VMAF_TIMEOUT_SECONDS,
        )
```

to:

```python
        timeout = _vmaf_timeout(target_width, target_height, duration_seconds)
        _log.debug("VMAF timeout: %ds (resolution=%dx%d, duration=%s)",
                    timeout, target_width, target_height, duration_seconds)
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_vmaf.py -v`
Expected: All 7 tests PASS

- [ ] **Step 7: Commit**

```bash
git add encoder/vmaf.py tests/test_vmaf.py
git commit -m "add dynamic VMAF timeout and multi-threaded libvmaf

Replace the fixed 600s timeout with a function that scales based on
resolution and sample duration. Enable n_threads=0 in the libvmaf
filter to use all available CPU cores."
```

---

### Task 2: Encoded-file seeking and dead parameter cleanup in `run_vmaf()`

**Files:**
- Modify: `encoder/vmaf.py:29-68` (function signature and command building)
- Test: `tests/test_vmaf.py` (add tests)

- [ ] **Step 1: Write failing tests for encoded-file seeking**

Append to `tests/test_vmaf.py`:

```python
from unittest.mock import patch, MagicMock
import subprocess

from encoder.vmaf import run_vmaf


def _make_vmaf_result(score: float) -> MagicMock:
    """Create a mock subprocess result that returns a VMAF score on stderr."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stderr = f"VMAF score: {score}".encode()
    mock.stdout = b""
    return mock


def test_run_vmaf_adds_encoded_seeking():
    with patch("encoder.vmaf.subprocess.run", return_value=_make_vmaf_result(92.5)) as mock_run:
        run_vmaf(
            ffmpeg_path="ffmpeg",
            source_path="/src/video.mkv",
            encoded_path="/enc/video.mp4",
            target_width=1920,
            target_height=1080,
            start_seconds=100.0,
            duration_seconds=15.0,
            encoded_start_seconds=52.5,
        )
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        # -ss for encoded file appears before the first -i
        ss_idx = cmd.index("-ss")
        first_i_idx = cmd.index("-i")
        assert ss_idx < first_i_idx
        assert cmd[ss_idx + 1] == "52.5"


def test_run_vmaf_no_encoded_seeking_by_default():
    with patch("encoder.vmaf.subprocess.run", return_value=_make_vmaf_result(92.5)) as mock_run:
        run_vmaf(
            ffmpeg_path="ffmpeg",
            source_path="/src/video.mkv",
            encoded_path="/enc/video.mp4",
            target_width=1920,
            target_height=1080,
        )
        cmd = mock_run.call_args[0][0]
        # No -ss should appear before the first -i (encoded input)
        first_i_idx = cmd.index("-i")
        pre_first_input = cmd[:first_i_idx]
        assert "-ss" not in pre_first_input


def test_run_vmaf_no_source_width_height_params():
    """Verify dead params were removed from signature."""
    import inspect
    sig = inspect.signature(run_vmaf)
    param_names = list(sig.parameters.keys())
    assert "source_width" not in param_names
    assert "source_height" not in param_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vmaf.py::test_run_vmaf_adds_encoded_seeking tests/test_vmaf.py::test_run_vmaf_no_encoded_seeking_by_default tests/test_vmaf.py::test_run_vmaf_no_source_width_height_params -v`
Expected: `test_run_vmaf_adds_encoded_seeking` fails (no `encoded_start_seconds` param), `test_run_vmaf_no_source_width_height_params` fails (params still exist)

- [ ] **Step 3: Add `encoded_start_seconds` param and remove dead params**

In `encoder/vmaf.py`, change the `run_vmaf` signature from:

```python
def run_vmaf(
    ffmpeg_path: str,
    source_path: str,
    encoded_path: str,
    source_width: int,  # noqa: ARG001 - reserved for future use
    source_height: int,  # noqa: ARG001 - reserved for future use
    target_width: int,
    target_height: int,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> dict | None:
```

to:

```python
def run_vmaf(
    ffmpeg_path: str,
    source_path: str,
    encoded_path: str,
    target_width: int,
    target_height: int,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    encoded_start_seconds: float | None = None,
) -> dict | None:
```

- [ ] **Step 4: Add encoded-file seeking to command building**

In `encoder/vmaf.py`, change the encoded-file input section from:

```python
    cmd: list[str] = [ffmpeg_path, "-hide_banner"]

    # Input args for encoded file (no seeking needed, it's already the segment)
    cmd.extend(["-i", encoded_path])
```

to:

```python
    cmd: list[str] = [ffmpeg_path, "-hide_banner"]

    if encoded_start_seconds is not None:
        cmd.extend(["-ss", str(encoded_start_seconds)])
    cmd.extend(["-i", encoded_path])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_vmaf.py -v`
Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add encoder/vmaf.py tests/test_vmaf.py
git commit -m "add encoded-file seeking and remove dead params from run_vmaf

Add encoded_start_seconds parameter to support seeking within the
encoded file for VMAF subsampling. Remove unused source_width and
source_height parameters."
```

---

### Task 3: Subsample window math in `_run_vmaf_scoring()`

**Files:**
- Modify: `encode.py:141-222` (`_run_vmaf_scoring` function)
- Test: `tests/test_vmaf.py` (add tests)

- [ ] **Step 1: Extract subsample math into a testable function**

The window calculation should be a standalone function in `encoder/vmaf.py` so it can be unit-tested without mocking the full scoring pipeline. Add to `encoder/vmaf.py` (after `VMAF_SAMPLE_SECONDS`):

```python
def vmaf_sample_window(
    source_start: float | None,
    segment_duration: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Compute the VMAF subsample window within a test encode segment.

    Returns (source_start, encoded_start, duration) for the VMAF command.
    """
    if segment_duration is None or segment_duration <= VMAF_SAMPLE_SECONDS:
        return source_start, None, segment_duration
    offset = (segment_duration - VMAF_SAMPLE_SECONDS) / 2
    vmaf_source_start = (source_start or 0) + offset
    return vmaf_source_start, offset, float(VMAF_SAMPLE_SECONDS)
```

- [ ] **Step 2: Write failing tests for `vmaf_sample_window`**

Append to `tests/test_vmaf.py`:

```python
from encoder.vmaf import vmaf_sample_window


def test_sample_window_long_segment():
    src_start, enc_start, dur = vmaf_sample_window(
        source_start=1350.0, segment_duration=120.0
    )
    assert dur == 15.0
    assert enc_start == 52.5
    assert src_start == 1402.5


def test_sample_window_short_segment():
    src_start, enc_start, dur = vmaf_sample_window(
        source_start=10.0, segment_duration=10.0
    )
    assert dur == 10.0
    assert enc_start is None
    assert src_start == 10.0


def test_sample_window_exact_15s():
    src_start, enc_start, dur = vmaf_sample_window(
        source_start=100.0, segment_duration=15.0
    )
    assert dur == 15.0
    assert enc_start is None
    assert src_start == 100.0


def test_sample_window_none_duration():
    src_start, enc_start, dur = vmaf_sample_window(
        source_start=None, segment_duration=None
    )
    assert dur is None
    assert enc_start is None
    assert src_start is None


def test_sample_window_none_source_start():
    src_start, enc_start, dur = vmaf_sample_window(
        source_start=None, segment_duration=120.0
    )
    assert dur == 15.0
    assert enc_start == 52.5
    assert src_start == 52.5
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_vmaf.py::test_sample_window_long_segment -v`
Expected: ImportError for `vmaf_sample_window`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vmaf.py -v`
Expected: All 15 tests PASS

- [ ] **Step 5: Update `_run_vmaf_scoring()` caller in `encode.py`**

In `encode.py`, change the `_run_vmaf_scoring` function. Update the import (line 149) from:

```python
    from encoder.vmaf import check_vmaf_support, run_vmaf, vmaf_quality_label
```

to:

```python
    from encoder.vmaf import check_vmaf_support, run_vmaf, vmaf_quality_label, vmaf_sample_window
```

Replace the seeking calculation and `run_vmaf` call (lines 176-207) from:

```python
        duration = src_info.get("duration", 0.0)
        start_seconds = None
        dur_seconds = None
        if duration > test_seconds:
            start_seconds = (duration - test_seconds) / 2
            dur_seconds = test_seconds

        # Get source and target dimensions
        src_w = src_info.get("video_width", 1280)
        src_h = src_info.get("video_height", 720)

        # Probe the encoded file for its actual resolution
        try:
            target_info = probe_file(result.output_path, ffprobe_path=ffprobe_path)
            tgt_w = target_info.get("video_width", src_w)
            tgt_h = target_info.get("video_height", src_h)
        except RuntimeError:
            tgt_w, tgt_h = 1280, 720

        console.print(f"  Scoring [cyan]{Path(result.source_path).name}[/cyan]...")

        scores = run_vmaf(
            ffmpeg_path=ffmpeg_path,
            source_path=result.source_path,
            encoded_path=result.output_path,
            source_width=src_w,
            source_height=src_h,
            target_width=tgt_w,
            target_height=tgt_h,
            start_seconds=start_seconds,
            duration_seconds=dur_seconds,
        )
```

to:

```python
        duration = src_info.get("duration", 0.0)
        start_seconds = None
        dur_seconds = None
        if duration > test_seconds:
            start_seconds = (duration - test_seconds) / 2
            dur_seconds = test_seconds

        vmaf_src_start, vmaf_enc_start, vmaf_dur = vmaf_sample_window(
            start_seconds, dur_seconds,
        )

        # Probe the encoded file for its actual resolution
        src_w = src_info.get("video_width", 1280)
        src_h = src_info.get("video_height", 720)
        try:
            target_info = probe_file(result.output_path, ffprobe_path=ffprobe_path)
            tgt_w = target_info.get("video_width", src_w)
            tgt_h = target_info.get("video_height", src_h)
        except RuntimeError:
            tgt_w, tgt_h = 1280, 720

        console.print(f"  Scoring [cyan]{Path(result.source_path).name}[/cyan]...")

        scores = run_vmaf(
            ffmpeg_path=ffmpeg_path,
            source_path=result.source_path,
            encoded_path=result.output_path,
            target_width=tgt_w,
            target_height=tgt_h,
            start_seconds=vmaf_src_start,
            duration_seconds=vmaf_dur,
            encoded_start_seconds=vmaf_enc_start,
        )
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest -v`
Expected: All tests PASS (including any existing tests that might reference `run_vmaf`)

- [ ] **Step 7: Commit**

```bash
git add encoder/vmaf.py encode.py tests/test_vmaf.py
git commit -m "subsample VMAF to 15 seconds from middle of test encode

Extract vmaf_sample_window() to compute the seek positions for both
the source and encoded files. The caller now scores only 15 seconds
instead of the full test segment, making VMAF feasible on high-res
content."
```

---

### Task 4: Progress spinner during VMAF scoring

**Files:**
- Modify: `encode.py:141-222` (`_run_vmaf_scoring` function)

- [ ] **Step 1: Add spinner around the VMAF call**

In `encode.py`, add `threading` to the imports at top of `_run_vmaf_scoring`. Replace the scoring section (the `console.print("  Scoring ...")` line through the `scores = run_vmaf(...)` call) from:

```python
        console.print(f"  Scoring [cyan]{Path(result.source_path).name}[/cyan]...")

        scores = run_vmaf(
            ffmpeg_path=ffmpeg_path,
            source_path=result.source_path,
            encoded_path=result.output_path,
            target_width=tgt_w,
            target_height=tgt_h,
            start_seconds=vmaf_src_start,
            duration_seconds=vmaf_dur,
            encoded_start_seconds=vmaf_enc_start,
        )
```

to:

```python
        scores: dict | None = None

        def _score() -> None:
            nonlocal scores
            scores = run_vmaf(
                ffmpeg_path=ffmpeg_path,
                source_path=result.source_path,
                encoded_path=result.output_path,
                target_width=tgt_w,
                target_height=tgt_h,
                start_seconds=vmaf_src_start,
                duration_seconds=vmaf_dur,
                encoded_start_seconds=vmaf_enc_start,
            )

        import threading
        import time

        worker = threading.Thread(target=_score)
        t0 = time.monotonic()
        worker.start()

        filename = Path(result.source_path).name
        with console.status("") as status:
            while worker.is_alive():
                elapsed = int(time.monotonic() - t0)
                mins, secs = divmod(elapsed, 60)
                status.update(f"  Scoring [cyan]{filename}[/cyan]... {mins}:{secs:02d}")
                worker.join(timeout=1.0)
```

- [ ] **Step 2: Verify the spinner works with a manual test**

This change is UI-only and cannot be meaningfully unit tested. Verify manually:

Run: `python encode.py -s <source_dir> -o <output_dir> --test-only --vmaf -p <preset>`

Expected:
- A spinner with elapsed time appears: `Scoring file.mp4... 0:12`
- The spinner stops when VMAF finishes
- The VMAF score table appears with a valid score (not "scoring failed")

- [ ] **Step 3: Run all existing tests to check for regressions**

Run: `python -m pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add encode.py
git commit -m "add spinner with elapsed time during VMAF scoring

Show a Rich spinner that updates every second while VMAF runs in a
background thread, so the CLI no longer appears frozen during scoring."
```

---

### Task 5: Final verification and cleanup

**Files:**
- Review: `encoder/vmaf.py`, `encode.py`, `tests/test_vmaf.py`

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -v`
Expected: All tests PASS

- [ ] **Step 2: Verify no references to removed params**

Run: `grep -rn "source_width\|source_height" encoder/ encode.py tests/`
Expected: No matches (the dead params have been removed everywhere)

- [ ] **Step 3: Verify no references to old timeout constant**

Run: `grep -rn "VMAF_TIMEOUT_SECONDS" encoder/ encode.py tests/`
Expected: No matches

- [ ] **Step 4: Manual Windows test**

This must be tested on the user's Windows machine with high-resolution content:

Run: `parallel-encode -s <source_with_4k_vr180> -o <output_dir> --test-only --vmaf`

Verify:
- Spinner appears with elapsed time
- VMAF completes without timeout
- Score and quality label display in the table
