"""Tests for VMAF scoring helpers."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from encoder.vmaf import _vmaf_timeout, VMAF_SAMPLE_SECONDS, run_vmaf


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
