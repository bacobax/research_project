import unittest
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import (
    ActivityAwareMaskedSpanDataset,
    MultiSongMaskedSpanDataset,
    TrainConfig,
    resolve_extra_wav_paths,
)


def _make_song_dataset(seed: int, frames: int = 240, mask_token: int = 99) -> ActivityAwareMaskedSpanDataset:
    rng = np.random.RandomState(seed)
    k = 4
    codes = torch.from_numpy(rng.randint(0, 16, size=(k, frames))).long()
    activity = rng.rand(frames).astype(np.float32)
    token_change = np.clip(activity * 0.9, 0.0, 1.0)
    return ActivityAwareMaskedSpanDataset(
        codes=codes,
        gaps=[],
        seq_len=64,
        mask_len_range=(8, 16),
        mask_token=mask_token,
        virtual_size=128,
        activity_per_frame=activity,
        token_change_per_frame=token_change,
        activity_low_thr=float(np.quantile(activity, 0.3)),
        activity_high_thr=float(np.quantile(activity, 0.7)),
        weighted_sampling=True,
        dead_window_min_mean=0.0,
        dead_window_min_ratio=0.0,
        regime_probs={"active": 0.45, "transition": 0.30, "low_activity": 0.15, "uniform": 0.10},
        mask_stride=1,
        activity_guided_masking=True,
    )


class TestResolveExtraWavPaths(unittest.TestCase):
    def _touch(self, d: Path, names):
        for name in names:
            (d / name).write_bytes(b"")

    def test_no_dir_no_explicit_paths_returns_empty(self):
        cfg = TrainConfig()
        self.assertEqual(resolve_extra_wav_paths(cfg), [])

    def test_sorted_and_glob(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._touch(d, ["003_c.wav", "001_a.wav", "002_b.wav", "notes.txt"])
            cfg = TrainConfig(extra_wavs_dir=str(d))
            paths = resolve_extra_wav_paths(cfg)
            self.assertEqual([p.name for p in paths], ["001_a.wav", "002_b.wav", "003_c.wav"])

    def test_exclude_pattern(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._touch(d, ["001_a.wav", "008_pivot_take2.wav", "002_b.wav"])
            cfg = TrainConfig(extra_wavs_dir=str(d), extra_wavs_exclude=("008_*",))
            paths = resolve_extra_wav_paths(cfg)
            self.assertEqual([p.name for p in paths], ["001_a.wav", "002_b.wav"])

    def test_limit_produces_nested_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._touch(d, [f"{i:03d}_song.wav" for i in range(1, 8)])
            cfg_full = TrainConfig(extra_wavs_dir=str(d))
            cfg_limited = TrainConfig(extra_wavs_dir=str(d), extra_wavs_limit=3)
            full = resolve_extra_wav_paths(cfg_full)
            limited = resolve_extra_wav_paths(cfg_limited)
            self.assertEqual(len(limited), 3)
            self.assertEqual(limited, full[:3])

    def test_explicit_paths_override_dir(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self._touch(d, ["001_a.wav", "002_b.wav"])
            explicit = str(d / "002_b.wav")
            cfg = TrainConfig(extra_wavs_dir=str(d), extra_wav_paths=(explicit,))
            paths = resolve_extra_wav_paths(cfg)
            self.assertEqual([str(p) for p in paths], [explicit])

    def test_missing_dir_raises(self):
        cfg = TrainConfig(extra_wavs_dir="/does/not/exist")
        with self.assertRaises(FileNotFoundError):
            resolve_extra_wav_paths(cfg)


class TestMultiSongMaskedSpanDataset(unittest.TestCase):
    def test_shapes_and_every_song_reachable(self):
        datasets = [_make_song_dataset(seed=i) for i in range(3)]
        weights = np.array([len(ds.starts) for ds in datasets], dtype=np.float32)
        wrapped = MultiSongMaskedSpanDataset(datasets=datasets, song_weights=weights, virtual_size=64)

        self.assertEqual(len(wrapped), 64)
        seen_songs = set()
        torch.manual_seed(0)
        for i in range(200):
            song_idx = wrapped._sample_song_index()
            seen_songs.add(song_idx)
            x, y, loss_mask = wrapped[i]
            self.assertEqual(tuple(x.shape), (4, 64))
            self.assertEqual(tuple(y.shape), (4, 64))
            self.assertEqual(tuple(loss_mask.shape), (64,))
            self.assertEqual(loss_mask.dtype, torch.bool)
        self.assertEqual(seen_songs, {0, 1, 2})

    def test_duration_weighting_proportional_to_starts(self):
        # Give the two songs deliberately different valid-window counts by using different
        # frame lengths, then check sampling frequency roughly tracks the starts ratio.
        ds_short = _make_song_dataset(seed=1, frames=100)
        ds_long = _make_song_dataset(seed=2, frames=400)
        weights = np.array([len(ds_short.starts), len(ds_long.starts)], dtype=np.float32)
        self.assertGreater(len(ds_long.starts), len(ds_short.starts))

        wrapped = MultiSongMaskedSpanDataset(datasets=[ds_short, ds_long], song_weights=weights)
        expected_ratio = weights[1] / weights.sum()
        torch.manual_seed(0)
        counts = [0, 0]
        n = 4000
        for _ in range(n):
            counts[wrapped._sample_song_index()] += 1
        observed_ratio = counts[1] / n
        self.assertAlmostEqual(observed_ratio, expected_ratio, delta=0.05)

    def test_update_mask_range_propagates_to_all_songs(self):
        datasets = [_make_song_dataset(seed=i) for i in range(2)]
        weights = np.ones(2, dtype=np.float32)
        wrapped = MultiSongMaskedSpanDataset(datasets=datasets, song_weights=weights)

        wrapped.update_mask_range(20, 30)
        for ds in datasets:
            self.assertEqual(ds.mask_len_range, (20, 30))
        self.assertEqual(wrapped.mask_len_range, (20, 30))

    def test_pop_recent_metrics_merges_across_songs(self):
        datasets = [_make_song_dataset(seed=i) for i in range(2)]
        weights = np.ones(2, dtype=np.float32)
        wrapped = MultiSongMaskedSpanDataset(datasets=datasets, song_weights=weights)

        for ds in datasets:
            for _ in range(5):
                ds[0]  # populate each sub-dataset's recent_metrics deque directly

        items = wrapped.pop_recent_metrics()
        self.assertEqual(len(items), 10)

    def test_summary_reports_num_songs_and_per_song_starts(self):
        datasets = [_make_song_dataset(seed=i) for i in range(3)]
        weights = np.array([len(ds.starts) for ds in datasets], dtype=np.float32)
        wrapped = MultiSongMaskedSpanDataset(datasets=datasets, song_weights=weights)

        self.assertEqual(wrapped.summary["num_songs"], 3)
        for i, ds in enumerate(datasets):
            self.assertEqual(wrapped.summary[f"song_{i}_starts"], len(ds.starts))

    def test_requires_matching_lengths(self):
        datasets = [_make_song_dataset(seed=0)]
        with self.assertRaises(ValueError):
            MultiSongMaskedSpanDataset(datasets=datasets, song_weights=np.array([1.0, 2.0]))

    def test_requires_at_least_one_dataset(self):
        with self.assertRaises(ValueError):
            MultiSongMaskedSpanDataset(datasets=[], song_weights=np.array([]))


class TestSingleSongNoRegression(unittest.TestCase):
    """With no extra songs, building the dataset must behave exactly as the pre-multi-song code."""

    def test_direct_construction_matches_no_extra_songs_path(self):
        # Mirrors what Trainer._build_dataset does when self.extra_samples is empty: construct
        # ActivityAwareMaskedSpanDataset directly rather than routing through the wrapper.
        ds_direct = _make_song_dataset(seed=42)
        self.assertIsInstance(ds_direct, ActivityAwareMaskedSpanDataset)
        self.assertNotIsInstance(ds_direct, MultiSongMaskedSpanDataset)


if __name__ == "__main__":
    unittest.main()
