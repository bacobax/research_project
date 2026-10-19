import hashlib
import tempfile
import unittest
import wave
from pathlib import Path

import yaml

from audio_infill import download_data as dd


def _write_wav(path: Path, sr=24000, ch=1, frames=100):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * frames * ch)


class DownloadConfigTests(unittest.TestCase):
    def test_repo_config_and_manifest_consistent(self):
        cfg = dd.load_config(dd.DEFAULT_CONFIG)
        entries = dd.parse_entries(cfg)
        self.assertEqual([e.index for e in entries], list(range(1, len(entries) + 1)))
        self.assertEqual(len({e.video_id for e in entries}), len(entries))
        pivot = cfg["pivot"]
        self.assertEqual(pivot["path"], "data/nuvole_bianche.mp3")
        self.assertEqual(len(pivot["sha256"]), 64)
        self.assertGreater(int(pivot["frames"]), 0)
        for w in cfg["weights"]:
            self.assertTrue(w["url"].startswith("https://"))
            self.assertEqual(len(w["sha256"]), 64)
            self.assertIn(w["sha256"][:8], w["filename"])
        manifest = dd.load_manifest(dd.DEFAULT_MANIFEST)
        for e in entries:
            self.assertIn((Path(cfg["output_dir"]) / e.filename).as_posix(), manifest)

    def test_duplicate_filenames_rejected(self):
        cfg = {"playlist": {"entries": [
            {"index": 1, "video_id": "a", "filename": "x.wav"},
            {"index": 2, "video_id": "b", "filename": "x.wav"}]}}
        with self.assertRaises(ValueError):
            dd.parse_entries(cfg)

    def test_wav_validation_and_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.wav"
            _write_wav(p)
            self.assertTrue(dd.wav_is_valid(p, 24000, 1))
            self.assertFalse(dd.wav_is_valid(p, 16000, 1))
            self.assertEqual(dd.sha256_file(p), hashlib.sha256(p.read_bytes()).hexdigest())
            bad = Path(d) / "bad.wav"
            bad.write_bytes(b"nope")
            self.assertFalse(dd.wav_is_valid(bad, 24000, 1))

    def test_update_manifest_then_check_and_idempotent_skip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "ds").mkdir()
            _write_wav(root / "ds" / "001_a.wav")
            cfg = {"output_dir": "ds", "playlist": {"sample_rate": 24000, "channels": 1, "entries": [
                {"index": 1, "video_id": "abc", "filename": "001_a.wav"}]}, "weights": []}
            mpath = root / "m.json"
            dd.update_manifest(cfg, root, mpath)
            manifest = dd.load_manifest(mpath)
            self.assertEqual(dd.check_against_manifest(Path("ds/001_a.wav"), manifest, root), "ok")
            _write_wav(root / "ds" / "001_a.wav", frames=7)
            self.assertEqual(dd.check_against_manifest(Path("ds/001_a.wav"), manifest, root), "mismatch")
            _write_wav(root / "ds" / "001_a.wav")
            # existing valid file => no network call, no failure
            rc = dd.download_playlist(cfg, manifest, root, strict=False, limit=None, dry_run=False)
            self.assertEqual(rc, 0)

    def test_verify_tolerates_drift_only_on_non_training_entries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "ds").mkdir()
            _write_wav(root / "ds" / "001_a.wav")
            _write_wav(root / "ds" / "002_b.wav")
            _write_wav(root / "pivot.wav")
            pivot = {"path": "pivot.wav", "sha256": dd.sha256_file(root / "pivot.wav"), "frames": 100, "sample_rate": 24000}
            cfg = {"output_dir": "ds", "playlist": {"sample_rate": 24000, "channels": 1, "entries": [
                {"index": 1, "video_id": "a", "filename": "001_a.wav"},
                {"index": 2, "video_id": "b", "filename": "002_b.wav", "training": False}]},
                "pivot": pivot, "weights": []}
            mpath = root / "m.json"
            dd.update_manifest(cfg, root, mpath)
            manifest = dd.load_manifest(mpath)
            _write_wav(root / "ds" / "002_b.wav", frames=7)  # drift on the non-training entry
            self.assertEqual(dd.verify(cfg, manifest, root), 0)
            self.assertEqual(dd.verify(cfg, manifest, root, strict=True), 1)
            _write_wav(root / "ds" / "001_a.wav", frames=7)  # drift on a training entry
            self.assertEqual(dd.verify(cfg, manifest, root), 1)

    def test_pivot_status_separates_reencode_from_wrong_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pivot.wav"
            _write_wav(p, frames=100)
            pivot = {"sha256": dd.sha256_file(p), "frames": 100, "sample_rate": 24000}
            self.assertEqual(dd.pivot_status(p, pivot), "ok")
            with wave.open(str(p), "wb") as wf:  # same length, different samples: a new encode
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
                wf.writeframes(b"\x01\x00" * 100)
            self.assertEqual(dd.pivot_status(p, pivot), "drift")
            _write_wav(p, frames=99)
            self.assertEqual(dd.pivot_status(p, pivot), "mismatch")
            self.assertEqual(dd.pivot_status(Path(d) / "absent.mp3", pivot), "missing")

    def test_cli_help(self):
        with self.assertRaises(SystemExit) as cm:
            dd.build_parser().parse_args(["--help"])
        self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
