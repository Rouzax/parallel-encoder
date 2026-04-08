"""Diagnose WebM seeking structure for audio sync issues.

Usage:
    python webm_seek_diag.py <file.webm>

Reports:
    - Stream info (codecs, bitrates, durations)
    - Number and average duration of clusters (ffprobe packets)
    - Audio packet density (gap between consecutive audio packets)
    - Whether cues appear near the start or end of the file
    - Opus codec_delay / initial_padding (pre-skip)
"""

import subprocess
import sys
import json
from pathlib import Path


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python webm_seek_diag.py <file.webm>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    file_size = path.stat().st_size
    print(f"File: {path}")
    print(f"Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MiB)")

    # ── Stream info ───────────────────────────────────────────────────
    section("1. STREAM INFO")
    out = run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ])
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        print("ffprobe failed:", out)
        sys.exit(1)

    for s in data.get("streams", []):
        ctype = s.get("codec_type", "?")
        cname = s.get("codec_name", "?")
        line = f"  {ctype}: {cname}"
        if ctype == "video":
            line += f" {s.get('width')}x{s.get('height')} @ {s.get('r_frame_rate')}fps"
        elif ctype == "audio":
            line += f" {s.get('sample_rate')}Hz {s.get('channels')}ch"
            initial = s.get("initial_padding")
            if initial is not None:
                line += f" pre-skip={initial}"
        print(line)

    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0))
    print(f"  duration: {duration:.2f}s")

    # ── Audio packet density ──────────────────────────────────────────
    section("2. AUDIO PACKET DENSITY (first 30s)")
    out = run([
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "packet=pts_time,pos,size",
        "-read_intervals", "0%+30",
        "-of", "csv=p=0",
        str(path),
    ])
    audio_pts = []
    for line in out.strip().splitlines():
        parts = line.split(",")
        if parts and parts[0]:
            try:
                audio_pts.append(float(parts[0]))
            except ValueError:
                pass

    if len(audio_pts) >= 2:
        print(f"  audio packets in first 30s: {len(audio_pts)}")
        print(f"  first packet pts: {audio_pts[0]:.4f}s")
        print(f"  last packet pts: {audio_pts[-1]:.4f}s")
        gaps = [audio_pts[i+1] - audio_pts[i] for i in range(len(audio_pts) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        print(f"  avg gap between audio packets: {avg_gap*1000:.1f}ms")
        print(f"  max gap: {max(gaps)*1000:.1f}ms")
    else:
        print("  not enough audio packets to analyse")

    # ── Cluster structure (via video keyframes as proxy) ──────────────
    section("3. VIDEO KEYFRAME / CLUSTER SPACING")
    out = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-read_intervals", "0%+60",
        "-of", "csv=p=0",
        str(path),
    ])
    keyframes = []
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                keyframes.append(float(parts[0]))
            except ValueError:
                pass

    print(f"  video keyframes in first 60s: {len(keyframes)}")
    if len(keyframes) >= 2:
        gaps = [keyframes[i+1] - keyframes[i] for i in range(len(keyframes) - 1)]
        print(f"  keyframe interval: avg={sum(gaps)/len(gaps):.2f}s min={min(gaps):.2f}s max={max(gaps):.2f}s")
    if keyframes:
        print(f"  first keyframe pts list: {[f'{k:.2f}' for k in keyframes[:10]]}")

    # ── Cues location (front vs back) ────────────────────────────────
    section("4. CUES LOCATION")
    # Read first 64KB and last 1MB looking for the Cues EBML ID (0x1c53bb6b)
    cues_id = bytes.fromhex("1c53bb6b")
    with path.open("rb") as f:
        head = f.read(131072)  # 128 KiB
        f.seek(-min(file_size, 1048576), 2)  # last 1 MiB
        tail = f.read()

    head_pos = head.find(cues_id)
    tail_pos = tail.find(cues_id)

    if head_pos >= 0:
        print(f"  Cues found in first 128KiB at offset {head_pos} - GOOD (cues_to_front working)")
    if tail_pos >= 0:
        actual_tail_pos = file_size - len(tail) + tail_pos
        print(f"  Cues found in last 1MiB at offset {actual_tail_pos} ({(actual_tail_pos/file_size)*100:.1f}% into file)")
    if head_pos < 0 and tail_pos < 0:
        print("  No Cues element found in head or tail - file may not be seekable")

    # ── Detect cluster_time_limit effectiveness ──────────────────────
    section("5. CLUSTER TIME LIMIT CHECK")
    # In WebM, audio gets a CuePoint per cluster ONLY if there's no video.
    # With video present, audio scans within the cluster. So cluster size
    # is the bound on audio scan delay.
    # Check video keyframe gaps - in our config, clusters break on keyframes
    # for video, and audio clusters break on cluster_time_limit (1000ms).
    if len(keyframes) >= 2:
        kf_gaps = [keyframes[i+1] - keyframes[i] for i in range(len(keyframes) - 1)]
        avg_kf_gap = sum(kf_gaps) / len(kf_gaps)
        if avg_kf_gap > 2.0:
            print(f"  WARNING: avg keyframe gap is {avg_kf_gap:.2f}s")
            print("  Video seek granularity is limited by keyframe interval.")
            print("  Consider tightening SVT-AV1 keyint (e.g. -svtav1-params keyint=2s)")
        else:
            print(f"  Video keyframe interval ({avg_kf_gap:.2f}s) looks fine")

    section("DONE")


if __name__ == "__main__":
    main()
