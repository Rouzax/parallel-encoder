# MKV AV1 Quality Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two MKV AV1 quality presets (CRF 22 and 24) with `tune=0`, plus a generic `svtav1_params` YAML field for flexible SVT-AV1 encoder parameter control.

**Architecture:** Extend the SVT-AV1 parameter building in `presets/loader.py` to merge an optional `video.svtav1_params` dict from YAML into the computed params (keyint, hierarchical-levels). Add two new presets to `config/presets.yaml`.

**Tech Stack:** Python, PyYAML, pytest

---

## File Map

- **Modify:** `presets/loader.py:227-249` - Refactor SVT-AV1 param building from list to dict, merge `svtav1_params`
- **Modify:** `config/presets.yaml` - Add two new preset entries
- **Modify:** `tests/test_presets.py` - Add tests for `svtav1_params` merging

---

### Task 1: Test svtav1_params merging in preset_to_ffmpeg_args

**Files:**
- Modify: `tests/test_presets.py`

- [ ] **Step 1: Write test for basic svtav1_params passthrough**

Add to `tests/test_presets.py`:

```python
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
```

- [ ] **Step 2: Write test for svtav1_params merging with keyint**

Add to `tests/test_presets.py`:

```python
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
```

- [ ] **Step 3: Write test for svtav1_params override of computed defaults**

Add to `tests/test_presets.py`:

```python
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
```

- [ ] **Step 4: Write test for svtav1_params with multiple params**

Add to `tests/test_presets.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `python -m pytest tests/test_presets.py::test_preset_to_ffmpeg_args_svtav1_params_basic tests/test_presets.py::test_preset_to_ffmpeg_args_svtav1_params_merges_with_keyint tests/test_presets.py::test_preset_to_ffmpeg_args_svtav1_params_overrides_computed tests/test_presets.py::test_preset_to_ffmpeg_args_svtav1_params_multiple -v`

Expected: All 4 tests FAIL because `svtav1_params` is not yet handled by the loader. The basic test will fail with `ValueError` on `args.index("-svtav1-params")` since no svtav1-params are emitted for MKV without keyint.

---

### Task 2: Implement svtav1_params merging in loader

**Files:**
- Modify: `presets/loader.py:227-249`

- [ ] **Step 1: Refactor SVT-AV1 param building to use dict and merge svtav1_params**

In `presets/loader.py`, replace lines 227-249 (the `if codec == "libsvtav1":` block) with:

```python
    if codec == "libsvtav1":
        svt_kv: dict[str, str] = {}

        keyint: int | None = video.get("keyint")
        if keyint is not None:
            svt_kv["keyint"] = f"{keyint}s"

        if container == "webm":
            svt_kv["hierarchical-levels"] = "3"

        for k, v in video.get("svtav1_params", {}).items():
            svt_kv[str(k)] = str(v)

        if svt_kv:
            svt_str = ":".join(f"{k}={v}" for k, v in svt_kv.items())
            args.extend(["-svtav1-params", svt_str])
```

This changes the internal representation from a list to a dict so that `svtav1_params` entries cleanly override computed defaults (e.g. if both `video.keyint` and `svtav1_params.keyint` are set, the dict key wins).

- [ ] **Step 2: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_presets.py::test_preset_to_ffmpeg_args_svtav1_params_basic tests/test_presets.py::test_preset_to_ffmpeg_args_svtav1_params_merges_with_keyint tests/test_presets.py::test_preset_to_ffmpeg_args_svtav1_params_overrides_computed tests/test_presets.py::test_preset_to_ffmpeg_args_svtav1_params_multiple -v`

Expected: All 4 PASS.

- [ ] **Step 3: Run full test suite to check for regressions**

Run: `python -m pytest -v`

Expected: All tests pass. The existing WebM preset tests (which rely on `keyint` and `hierarchical-levels` being emitted) should still work because the dict approach preserves the same output format.

- [ ] **Step 4: Commit**

```bash
git add presets/loader.py tests/test_presets.py
git commit -m "add generic svtav1_params support to preset loader

Refactors SVT-AV1 parameter building from list to dict so that
YAML-specified svtav1_params merge cleanly with computed defaults
(keyint, hierarchical-levels). Explicit params override computed
ones on conflict."
```

---

### Task 3: Add the two new presets to presets.yaml

**Files:**
- Modify: `config/presets.yaml`

- [ ] **Step 1: Add the new preset section**

In `config/presets.yaml`, after the existing native-resolution H.265 presets section (after line 103, the `subtitles: all` of `mkv-h265-10bit-medium-cq30-all-audio`), add:

```yaml

  # ─────────────────────────────────────────────────────────────
  # AV1 (SVT-AV1) — native resolution MKV (quality-focused)
  # ─────────────────────────────────────────────────────────────
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

- [ ] **Step 2: Verify presets load without errors**

Run: `python -c "from presets.loader import load_presets; p = load_presets('config/presets.yaml'); print(f'{len(p)} presets loaded'); assert 'mkv-av1-p6-crf22-all-audio' in p; assert 'mkv-av1-p6-crf24-all-audio' in p; print('Both new presets found')"`

Expected:
```
(total count) presets loaded
Both new presets found
```

- [ ] **Step 3: Verify FFmpeg args for the new presets**

Run: `python -c "
from presets.loader import load_presets, preset_to_ffmpeg_args
presets = load_presets('config/presets.yaml')
source = {'video_width': 3840, 'video_height': 2160, 'audio_streams': [{'codec': 'opus', 'language': 'eng', 'channels': '2'}]}
for key in ('mkv-av1-p6-crf22-all-audio', 'mkv-av1-p6-crf24-all-audio'):
    args = preset_to_ffmpeg_args(presets[key], source)
    print(f'{key}:')
    print(f'  {\" \".join(args)}')
    assert '-svtav1-params' in args
    idx = args.index('-svtav1-params')
    assert 'tune=0' in args[idx + 1]
    print(f'  svtav1-params: {args[idx + 1]}')
    print()
"`

Expected: Both presets emit `-svtav1-params tune=0` with no scaling filter.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add config/presets.yaml
git commit -m "add MKV AV1 quality presets (CRF 22 and 24)

Native resolution, SVT-AV1 preset 6 with tune=0 (SSIM-optimized).
Quality-focused presets for 4K HDR content with audio passthrough."
```
