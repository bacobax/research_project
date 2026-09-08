#!/usr/bin/env python3
"""Download a YouTube playlist and convert each track to a standardized WAV.

Run with:
    uv run python scripts/download_playlist_dataset.py

Processes playlist entries strictly one at a time to minimize peak disk
usage: download source -> MP3 -> verify -> delete source -> WAV -> verify
-> delete MP3 -> next entry. Never downloads more than one source file at
once, and the output directory ends up containing only the finished WAVs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yt_dlp

PLAYLIST_URL = "https://youtube.com/playlist?list=OLAK5uy_ld5G3KIUyG3vyVf7wDhdmRExJIf0u1fzw&si=y3yuqq7WVrgaIofH"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "dataset"

TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_SAMPLE_FMT = "s16"  # signed 16-bit PCM

MAX_TITLE_LEN = 80


@dataclass
class PlaylistEntry:
    index: int  # 1-based, stable playlist position
    video_id: str | None
    title: str


def sanitize_title(title: str) -> str:
    """Make a title safe to use as a filename component."""
    title = title.strip()
    # Replace filesystem/shell-hostile characters with underscores.
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)
    # Collapse whitespace to single underscores.
    title = re.sub(r"\s+", "_", title)
    # Keep only a conservative character set.
    title = re.sub(r"[^A-Za-z0-9._-]", "_", title)
    # Collapse repeated underscores and strip leading/trailing junk.
    title = re.sub(r"_+", "_", title).strip("._-")
    if not title:
        title = "untitled"
    return title[:MAX_TITLE_LEN]


def index_width(total: int) -> int:
    return max(3, len(str(total)))


def final_wav_name(index: int, width: int, title: str) -> str:
    return f"{index:0{width}d}_{sanitize_title(title)}.wav"


def run_ffmpeg(args: list[str], description: str) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed: {result.stderr.strip()}")


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def verify_wav_standardized(path: Path) -> bool:
    """Confirm the WAV header matches the required format without loading audio."""
    import wave

    try:
        with wave.open(str(path), "rb") as wf:
            return (
                wf.getnchannels() == TARGET_CHANNELS
                and wf.getframerate() == TARGET_SAMPLE_RATE
                and wf.getsampwidth() == 2  # 16-bit
                and wf.getnframes() > 0
            )
    except (wave.Error, EOFError, OSError):
        return False


def cleanup_prefix(directory: Path, prefix: str, keep: Path | None = None) -> None:
    """Remove all files starting with prefix in directory, except `keep`."""
    for path in directory.glob(f"{prefix}.*"):
        if keep is not None and path.resolve() == keep.resolve():
            continue
        try:
            path.unlink()
        except OSError:
            pass
    # yt-dlp/ffmpeg temp artifacts (.part, .ytdl, .part-Frag*) share the same stem.
    for path in directory.glob(f"{prefix}*.part*"):
        try:
            path.unlink()
        except OSError:
            pass
    for path in directory.glob(f"{prefix}*.ytdl"):
        try:
            path.unlink()
        except OSError:
            pass


def fetch_playlist_entries(playlist_url: str) -> list[PlaylistEntry]:
    ydl_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    raw_entries = info.get("entries") or []
    entries: list[PlaylistEntry] = []
    for i, entry in enumerate(raw_entries, start=1):
        if entry is None:
            entries.append(PlaylistEntry(index=i, video_id=None, title=f"unavailable_{i}"))
            continue
        video_id = entry.get("id")
        title = entry.get("title") or f"untitled_{i}"
        entries.append(PlaylistEntry(index=i, video_id=video_id, title=title))
    return entries


def download_source(video_id: str, dest_prefix: Path) -> Path:
    """Download the best available audio-only source for a single video."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{dest_prefix}.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "writethumbnail": False,
        "writeinfojson": False,
        "writedescription": False,
        "writeannotations": False,
        "postprocessors": [],
        "retries": 3,
        "noprogress": True,
    }
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    candidates = sorted(dest_prefix.parent.glob(f"{dest_prefix.name}.*"))
    # Exclude any stray temp files if present.
    candidates = [c for c in candidates if not c.name.endswith((".part", ".ytdl"))]
    if not candidates:
        raise RuntimeError("download completed but no source file was found")
    return candidates[0]


def process_entry(entry: PlaylistEntry, width: int, total: int) -> str:
    """Returns one of: 'success', 'skipped', 'failed'."""
    prefix_name = f"{entry.index:0{width}d}"
    final_wav = OUTPUT_DIR / final_wav_name(entry.index, width, entry.title)
    header = f"[{entry.index:0{width}d}/{total}] {entry.title}"

    if is_nonempty_file(final_wav) and verify_wav_standardized(final_wav):
        print(f"{header} — already exists, skipping")
        return "skipped"

    print(header)

    if entry.video_id is None:
        print("  Error: playlist entry unavailable (private/deleted video). Skipping.")
        return "failed"

    prefix_path = OUTPUT_DIR / prefix_name
    source_path: Path | None = None
    mp3_path = OUTPUT_DIR / f"{prefix_name}.mp3"

    try:
        print("  Downloading...")
        source_path = download_source(entry.video_id, prefix_path)

        print("  Creating MP3...")
        run_ffmpeg(
            ["-i", str(source_path), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(mp3_path)],
            "MP3 conversion",
        )
        if not is_nonempty_file(mp3_path):
            raise RuntimeError("MP3 file missing or empty after conversion")

        source_path.unlink(missing_ok=True)
        source_path = None
        print("  Source removed.")

        print("  Converting MP3 → WAV...")
        tmp_wav = OUTPUT_DIR / f"{prefix_name}.wav.tmp"
        run_ffmpeg(
            [
                "-i", str(mp3_path),
                "-ac", str(TARGET_CHANNELS),
                "-ar", str(TARGET_SAMPLE_RATE),
                "-acodec", "pcm_s16le",
                "-f", "wav",
                str(tmp_wav),
            ],
            "WAV conversion",
        )
        if not is_nonempty_file(tmp_wav) or not verify_wav_standardized(tmp_wav):
            raise RuntimeError("WAV file missing, empty, or not in the expected format")

        tmp_wav.rename(final_wav)
        print(f"  WAV verified: {final_wav.relative_to(OUTPUT_DIR.parent)}")

        mp3_path.unlink(missing_ok=True)
        print("  Intermediate MP3 removed.")
        print("  Done.")
        return "success"

    except Exception as exc:  # noqa: BLE001 - report and continue with next song
        print(f"  ERROR [{entry.index}/{total}] '{entry.title}': {exc}")
        return "failed"

    finally:
        # Never delete an already-valid final WAV; only clean this entry's
        # intermediates/temp files.
        cleanup_prefix(OUTPUT_DIR, prefix_name, keep=final_wav if final_wav.exists() else None)


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install it (e.g. `sudo apt install -y ffmpeg`) and retry.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching playlist entries from: {PLAYLIST_URL}")
    entries = fetch_playlist_entries(PLAYLIST_URL)
    total = len(entries)
    if total == 0:
        print("Playlist is empty or could not be read.")
        return 1
    width = index_width(total)
    print(f"Found {total} playlist entries. Output directory: {OUTPUT_DIR}\n")

    counts = {"success": 0, "skipped": 0, "failed": 0}
    for entry in entries:
        result = process_entry(entry, width, total)
        counts[result] += 1
        print()

    print("Completed playlist processing.")
    print(f"Successful: {counts['success']}")
    print(f"Skipped:    {counts['skipped']}")
    print(f"Failed:     {counts['failed']}")
    print(f"Total:      {total}")

    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
