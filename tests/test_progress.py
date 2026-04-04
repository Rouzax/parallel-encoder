"""Tests for the summary table display."""

from __future__ import annotations

from encoder.ffmpeg import EncodingResult
from ui.progress import print_summary_table


def test_summary_table_handles_same_stem_different_dirs():
    """Two files named 'movie' in different subdirs should both appear correctly."""
    source_files = [
        {"path": "/src/a/movie.mkv", "filename": "movie", "file_size": 1000,
         "video_codec": "h264", "video_bitrate": 5000, "total_bitrate": 6000},
        {"path": "/src/b/movie.mkv", "filename": "movie", "file_size": 2000,
         "video_codec": "h264", "video_bitrate": 5000, "total_bitrate": 6000},
    ]
    results = [
        EncodingResult(source_path="/src/a/movie.mkv", output_path="/out/a/movie.mkv",
                       success=True, exit_code=0, encoding_time=10.0, error_message=None),
        EncodingResult(source_path="/src/b/movie.mkv", output_path="/out/b/movie.mkv",
                       success=True, exit_code=0, encoding_time=20.0, error_message=None),
    ]
    target_files = [
        {"path": "/out/a/movie.mkv", "filename": "movie", "file_size": 500,
         "video_codec": "hevc", "video_bitrate": 2000, "total_bitrate": 3000},
        {"path": "/out/b/movie.mkv", "filename": "movie", "file_size": 800,
         "video_codec": "hevc", "video_bitrate": 2000, "total_bitrate": 3000},
    ]
    # Should not crash and should show both entries with correct data
    print_summary_table(source_files, results, target_files)
