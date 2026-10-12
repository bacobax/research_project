"""Fetch every internet-sourced artifact (playlist audio, EnCodec weights) reproducibly.

Usage (from the repository root)::

    uv run audio-infill-download                 # playlist WAVs + EnCodec weights
    uv run audio-infill-download --only weights  # just the pretrained EnCodec checkpoint
    uv run audio-infill-download --verify        # offline: check files against the manifest
    uv run audio-infill-download --update-manifest   # (re)write checksums for files on disk

Sources (YouTube video IDs, weight URLs, output filenames) are pinned in
``configs/data/download.yaml``; reference checksums live in
``configs/data/dataset_manifest.json``. The run is idempotent: valid existing files are
skipped, and each entry is processed one at a time (download -> MP3 -> WAV -> delete
intermediates) to bound disk usage.

Determinism caveat: YouTube may re-encode or remove videos, so a fresh download is not
guaranteed to be bit-identical. Weights are checksum-verified (hard failure on mismatch);
audio checksum mismatches are reported as warnings unless ``--strict`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/data/download.yaml")
DEFAULT_MANIFEST = Path("configs/data/dataset_manifest.json")
TARGET_SAMPLE_FMT_BYTES = 2  # 16-bit PCM


@dataclass
class PlaylistEntry:
    """One playlist item: position, YouTube video id and output filename."""
    index: int
    video_id: str
    filename: str
    training: bool = True


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the download YAML."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("output_dir", "playlist", "weights"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required key '{key}'")
    return cfg


def parse_entries(cfg: dict[str, Any]) -> list[PlaylistEntry]:
    """Build PlaylistEntry objects from the config, rejecting duplicate filenames."""
    entries = [
        PlaylistEntry(
            index=int(e["index"]),
            video_id=str(e["video_id"]),
            filename=str(e["filename"]),
            training=bool(e.get("training", True)),
        )
        for e in cfg["playlist"]["entries"]
    ]
    if len({e.filename for e in entries}) != len(entries):
        raise ValueError("duplicate filenames in playlist entries")
    return entries


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load the checksum manifest JSON (empty dict if absent)."""
    if not path.is_file():
        return {}
    with open(path) as f:
        return json.load(f).get("files", {})


def wav_info(path: Path) -> dict[str, int] | None:
    """Header info for a WAV, or None if unreadable."""
    try:
        with wave.open(str(path), "rb") as wf:
            return {
                "channels": wf.getnchannels(),
                "sample_rate": wf.getframerate(),
                "sample_width": wf.getsampwidth(),
                "frames": wf.getnframes(),
            }
    except (wave.Error, EOFError, OSError):
        return None


def wav_is_valid(path: Path, sample_rate: int, channels: int) -> bool:
    info = wav_info(path)
    return (
        info is not None
        and info["channels"] == channels
        and info["sample_rate"] == sample_rate
        and info["sample_width"] == TARGET_SAMPLE_FMT_BYTES
        and info["frames"] > 0
    )


def run_ffmpeg(args: list[str], description: str) -> None:
    """Run ffmpeg with quiet flags; raise RuntimeError with stderr on failure."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed: {result.stderr.strip()}")


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def download_source(video_id: str, dest_prefix: Path) -> Path:
    """Download the best audio-only source for one video via yt-dlp."""
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{dest_prefix}.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    candidates = [
        c for c in sorted(dest_prefix.parent.glob(f"{dest_prefix.name}.*"))
        if not c.name.endswith((".part", ".ytdl"))
    ]
    if not candidates:
        raise RuntimeError("download completed but no source file was found")
    return candidates[0]


def _cleanup(directory: Path, prefix: str) -> None:
    for path in directory.glob(f"{prefix}.*"):
        if path.suffix != ".wav":
            path.unlink(missing_ok=True)
    for path in directory.glob(f"{prefix}*.part*"):
        path.unlink(missing_ok=True)


def fetch_entry(entry: PlaylistEntry, out_dir: Path, sample_rate: int, channels: int) -> str:
    """Return 'success' or 'failed' for one playlist entry (caller handles skipping)."""
    prefix = f"_tmp_{entry.index:03d}"
    final = out_dir / entry.filename
    mp3 = out_dir / f"{prefix}.mp3"
    tmp_wav = out_dir / f"{prefix}.wav.part"
    try:
        src = download_source(entry.video_id, out_dir / prefix)
        run_ffmpeg(["-i", str(src), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(mp3)], "MP3 conversion")
        if not _nonempty(mp3):
            raise RuntimeError("MP3 missing or empty after conversion")
        src.unlink(missing_ok=True)
        run_ffmpeg(
            ["-i", str(mp3), "-ac", str(channels), "-ar", str(sample_rate),
             "-acodec", "pcm_s16le", "-f", "wav", str(tmp_wav)],
            "WAV conversion",
        )
        if not wav_is_valid(tmp_wav, sample_rate, channels):
            raise RuntimeError("WAV missing, empty, or not in the expected format")
        tmp_wav.replace(final)
        return "success"
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"  ERROR [{entry.index}] {entry.filename}: {exc}")
        return "failed"
    finally:
        _cleanup(out_dir, prefix)
        tmp_wav.unlink(missing_ok=True)


def check_against_manifest(path: Path, manifest: dict[str, dict[str, Any]], root: Path = Path(".")) -> str:
    """'ok', 'mismatch', or 'unpinned' for a file relative to the manifest."""
    ref = manifest.get(path.as_posix())
    if ref is None:
        return "unpinned"
    return "ok" if sha256_file(root / path) == ref["sha256"] else "mismatch"


def download_playlist(cfg, manifest, root: Path, strict: bool, limit: int | None, dry_run: bool) -> int:
    """Fetch missing playlist WAVs; return the number of failures."""
    pl = cfg["playlist"]
    sr, ch = int(pl["sample_rate"]), int(pl["channels"])
    out_dir = root / cfg["output_dir"]
    entries = parse_entries(cfg)[:limit]
    if not dry_run:
        if shutil.which("ffmpeg") is None:
            print("ERROR: ffmpeg not found on PATH (e.g. `sudo apt install -y ffmpeg`).")
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
    failed = warned = 0
    for e in entries:
        final = out_dir / e.filename
        label = f"[{e.index:03d}/{len(entries)}] {e.filename}"
        if final.is_file() and wav_is_valid(final, sr, ch):
            print(f"{label}: exists, skipping")
        elif dry_run:
            print(f"{label}: would download https://www.youtube.com/watch?v={e.video_id}")
            continue
        else:
            print(f"{label}: downloading")
            if fetch_entry(e, out_dir, sr, ch) == "failed":
                failed += 1
                continue
        status = check_against_manifest(Path(cfg["output_dir"]) / e.filename, manifest, root)
        if status == "mismatch":
            warned += 1
            print("  WARNING: sha256 differs from manifest (upstream re-encode or changed video)")
    print(f"Playlist: {len(entries) - failed} ok, {failed} failed, {warned} checksum warnings")
    return 1 if failed or (strict and warned) else 0


def fetch_weights(cfg, root: Path, dry_run: bool) -> int:
    """Download the EnCodec weights into the torch hub cache, verifying sha256; return failures."""
    import torch

    dest_dir = Path(torch.hub.get_dir()) / "checkpoints"
    rc = 0
    for w in cfg["weights"]:
        dest = dest_dir / w["filename"]
        if dest.is_file() and sha256_file(dest) == w["sha256"]:
            print(f"weights {w['name']}: present and verified ({dest})")
            continue
        if dry_run:
            print(f"weights {w['name']}: would download {w['url']} -> {dest}")
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        print(f"weights {w['name']}: downloading {w['url']}")
        urllib.request.urlretrieve(w["url"], part)
        if sha256_file(part) != w["sha256"]:
            part.unlink(missing_ok=True)
            print(f"ERROR: checksum mismatch for {w['name']}")
            rc = 1
            continue
        part.replace(dest)
        print(f"weights {w['name']}: verified -> {dest}")
    return rc


def verify(cfg, manifest, root: Path) -> int:
    """Offline check of every expected file against the pinned checksums."""
    import torch

    bad = 0
    for e in parse_entries(cfg):
        rel = Path(cfg["output_dir"]) / e.filename
        path = root / rel
        if not path.is_file():
            print(f"MISSING   {rel}")
            bad += 1
            continue
        st = check_against_manifest(rel, manifest, root)
        print(f"{st.upper():9s} {rel}")
        bad += st == "mismatch"
    for m in cfg.get("manual", []):
        path = root / m["path"]
        if not path.is_file():
            print(f"MISSING   {m['path']}  (manual input; see configs/data/download.yaml)")
            bad += 1
        elif sha256_file(path) != m["sha256"]:
            print(f"MISMATCH  {m['path']}")
            bad += 1
        else:
            print(f"OK        {m['path']}")
    for w in cfg["weights"]:
        path = Path(torch.hub.get_dir()) / "checkpoints" / w["filename"]
        if not path.is_file():
            print(f"MISSING   {path}")
            bad += 1
        elif sha256_file(path) != w["sha256"]:
            print(f"MISMATCH  {path}")
            bad += 1
        else:
            print(f"OK        {path}")
    print("verify:", "all good" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def update_manifest(cfg, root: Path, manifest_path: Path) -> int:
    """Rewrite the manifest from the WAVs currently on disk; return 0."""
    files: dict[str, dict[str, Any]] = {}
    for e in parse_entries(cfg):
        rel = Path(cfg["output_dir"]) / e.filename
        path = root / rel
        if not path.is_file():
            print(f"skip (missing): {rel}")
            continue
        info = wav_info(path) or {}
        files[rel.as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "frames": info.get("frames"),
            "video_id": e.video_id,
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"files": dict(sorted(files.items()))}, f, indent=2)
        f.write("\n")
    print(f"wrote {len(files)} entries -> {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(
        description="Download playlist audio and pretrained EnCodec weights (idempotent, "
        "checksum-verified). Sources are pinned in configs/data/download.yaml.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Source config YAML")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Checksum manifest JSON")
    p.add_argument("--only", choices=["all", "playlist", "weights"], default="all",
                   help="Which artifacts to fetch (default: all)")
    p.add_argument("--verify", action="store_true", help="Offline: check files against checksums, no network")
    p.add_argument("--update-manifest", action="store_true",
                   help="Write checksums of the files currently on disk to the manifest")
    p.add_argument("--strict", action="store_true", help="Treat audio checksum mismatches as errors")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N playlist entries")
    p.add_argument("--dry-run", action="store_true", help="Print what would be downloaded")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (audio-infill-download); returns the process exit code."""
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    cfg = load_config(args.config)
    manifest = load_manifest(args.manifest)
    if args.verify:
        return verify(cfg, manifest, root)
    if args.update_manifest:
        return update_manifest(cfg, root, args.manifest)
    rc = 0
    if args.only in ("all", "weights"):
        rc |= fetch_weights(cfg, root, args.dry_run)
    if args.only in ("all", "playlist"):
        rc |= download_playlist(cfg, manifest, root, args.strict, args.limit, args.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
