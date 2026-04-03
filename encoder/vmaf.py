"""VMAF quality scoring for encoded video files."""

from __future__ import annotations

import json
import logging
import re
import subprocess

_log = logging.getLogger("parallel-encoder")

VMAF_TIMEOUT_SECONDS = 600  # 10 minutes per comparison


def check_vmaf_support(ffmpeg_path: str) -> bool:
    """Check if the ffmpeg build includes libvmaf."""
    try:
        result = subprocess.run(
            [ffmpeg_path, "-filters"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "libvmaf" in result.stdout
    except Exception:
        return False


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
    """Run VMAF comparison between source and encoded file.

    The source is scaled to match the encoded resolution before comparison.

    Returns a dict with vmaf/psnr scores, or None on failure.
    """
    # Build the filter: scale source down to encoded resolution, then compare
    # [0:v] = encoded (distorted), [1:v] = source (reference)
    # VMAF expects: distorted first, reference second
    scale_filter = f"scale={target_width}:{target_height}:flags=bicubic"
    vmaf_filter = (
        f"[1:v]{scale_filter}[ref];"
        f"[0:v][ref]libvmaf=log_fmt=json:log_path=-"
    )

    cmd: list[str] = [ffmpeg_path, "-hide_banner"]

    # Input args for encoded file (no seeking needed, it's already the segment)
    cmd.extend(["-i", encoded_path])

    # Input args for source file (seek to same segment)
    if start_seconds is not None:
        cmd.extend(["-ss", str(start_seconds)])
    if duration_seconds is not None:
        cmd.extend(["-t", str(duration_seconds)])
    cmd.extend(["-i", source_path])

    cmd.extend(["-lavfi", vmaf_filter, "-f", "null", "-"])

    _log.debug("VMAF command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=VMAF_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        _log.warning("VMAF timed out for %s", encoded_path)
        return None

    stderr_text = result.stderr.decode("utf-8", errors="replace")

    if result.returncode != 0:
        _log.warning("VMAF failed for %s: %s", encoded_path, stderr_text[-500:])
        return None

    # Parse VMAF score from stderr (ffmpeg prints it there)
    # Look for: "VMAF score: 93.123456"
    vmaf_match = re.search(r"VMAF score:\s*([\d.]+)", stderr_text)
    if vmaf_match:
        return {"vmaf": float(vmaf_match.group(1))}

    # Try parsing JSON from stdout (log_path=-)
    stdout_text = result.stdout.decode("utf-8", errors="replace")
    try:
        data = json.loads(stdout_text)
        pooled = data.get("pooled_metrics", {})
        vmaf_score = pooled.get("vmaf", {}).get("mean")
        if vmaf_score is not None:
            return {"vmaf": vmaf_score}
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: parse from stderr log lines
    # Some ffmpeg versions print per-frame then a summary
    vmaf_mean = re.search(r"vmaf.*?mean:\s*([\d.]+)", stderr_text, re.IGNORECASE)
    if vmaf_mean:
        return {"vmaf": float(vmaf_mean.group(1))}

    _log.warning("Could not parse VMAF score from output for %s", encoded_path)
    _log.debug("VMAF stderr: %s", stderr_text[-1000:])
    return None


def vmaf_quality_label(score: float) -> str:
    """Return a human-readable quality label for a VMAF score."""
    if score >= 95:
        return "excellent"
    if score >= 90:
        return "very good"
    if score >= 80:
        return "good"
    if score >= 70:
        return "acceptable"
    return "poor"
