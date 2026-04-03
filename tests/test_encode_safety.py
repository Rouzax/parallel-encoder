"""Tests for output directory safety in encode.py."""

from __future__ import annotations

from pathlib import Path

from encode import _cleanup_test_outputs


def test_cleanup_test_outputs_only_removes_listed_files(tmp_path):
    """Cleanup after test encode should only remove files we created."""
    # Simulate pre-existing file
    existing = tmp_path / "important_file.txt"
    existing.write_text("do not delete")

    # Simulate test encode outputs
    test_out_1 = tmp_path / "video1.mkv"
    test_out_2 = tmp_path / "video2.mkv"
    test_out_1.write_text("test output 1")
    test_out_2.write_text("test output 2")

    _cleanup_test_outputs([str(test_out_1), str(test_out_2)])

    assert existing.exists(), "Pre-existing file was deleted!"
    assert not test_out_1.exists()
    assert not test_out_2.exists()


def test_cleanup_test_outputs_handles_missing_files(tmp_path):
    """Cleanup should not crash if a file is already gone."""
    _cleanup_test_outputs([str(tmp_path / "nonexistent.mkv")])
