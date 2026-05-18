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
