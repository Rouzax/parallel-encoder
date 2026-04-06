"""Tests for sidecar file copying logic."""

from __future__ import annotations

from pathlib import Path

from encode import _collect_video_stems, _copy_sidecars_for_file

VIDEO_EXTS = (".mkv", ".mp4", ".webm", ".avi", ".ts")


def _setup_source(tmp_path: Path) -> tuple[Path, Path]:
    """Create a source tree with two videos and their sidecars."""
    source = tmp_path / "source" / "Shows"
    source.mkdir(parents=True)
    output = tmp_path / "output" / "Shows"
    output.mkdir(parents=True)

    # Two video files
    (source / "Movie A.mkv").write_bytes(b"video-a")
    (source / "Movie B.mkv").write_bytes(b"video-b")

    # Per-video sidecars
    (source / "Movie A-fanart.jpg").write_bytes(b"fanart-a")
    (source / "Movie A-poster.jpg").write_bytes(b"poster-a")
    (source / "Movie A.nfo").write_bytes(b"nfo-a")
    (source / "Movie B-fanart.jpg").write_bytes(b"fanart-b")
    (source / "Movie B-poster.jpg").write_bytes(b"poster-b")
    (source / "Movie B.nfo").write_bytes(b"nfo-b")

    # Directory-level sidecar (not tied to any video)
    (source / "folder.jpg").write_bytes(b"folder")

    return source, output


def test_only_copies_matching_sidecars(tmp_path: Path) -> None:
    """When Movie A finishes, only Movie A sidecars should be copied."""
    source, output = _setup_source(tmp_path)
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"

    _copy_sidecars_for_file(
        source / "Movie A.mkv", source_root, output_root, VIDEO_EXTS,
    )

    # Movie A sidecars should be present
    assert (output / "Movie A-fanart.jpg").exists()
    assert (output / "Movie A-poster.jpg").exists()
    assert (output / "Movie A.nfo").exists()

    # Movie B sidecars must NOT be present
    assert not (output / "Movie B-fanart.jpg").exists()
    assert not (output / "Movie B-poster.jpg").exists()
    assert not (output / "Movie B.nfo").exists()


def test_directory_level_sidecar_copied(tmp_path: Path) -> None:
    """Directory-level files like folder.jpg should be copied."""
    source, output = _setup_source(tmp_path)
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"

    _copy_sidecars_for_file(
        source / "Movie A.mkv", source_root, output_root, VIDEO_EXTS,
    )

    assert (output / "folder.jpg").exists()


def test_no_duplicate_copy(tmp_path: Path) -> None:
    """Calling twice should not re-copy already existing files."""
    source, output = _setup_source(tmp_path)
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"

    first = _copy_sidecars_for_file(
        source / "Movie A.mkv", source_root, output_root, VIDEO_EXTS,
    )
    second = _copy_sidecars_for_file(
        source / "Movie A.mkv", source_root, output_root, VIDEO_EXTS,
    )

    assert first > 0
    assert second == 0


def test_collect_video_stems(tmp_path: Path) -> None:
    source, _ = _setup_source(tmp_path)
    stems = _collect_video_stems(source, VIDEO_EXTS)
    assert stems == {"Movie A", "Movie B"}
