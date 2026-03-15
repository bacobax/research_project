import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import ActivityAwareMaskedSpanDataset, build_fixed_validation_examples, is_non_gap_window


class TestValidationPoolBuilder(unittest.TestCase):
    def setUp(self):
        self.codes = torch.randint(low=0, high=8, size=(3, 160), dtype=torch.long)
        self.activity = np.zeros(160, dtype=np.float32)
        self.activity[70:120] = 0.95
        self.gaps = [(125, 140)]

    def test_builds_deterministic_band_specific_examples(self):
        pool_a, holdout_a = build_fixed_validation_examples(
            codes=self.codes,
            gaps=self.gaps,
            seq_len=32,
            mask_len_range=(8, 12),
            mask_token=99,
            activity_per_frame=self.activity,
            activity_low_thr=0.01,
            activity_high_thr=0.80,
            examples_per_band=6,
            mask_stride=1,
            seed=123,
            sample_name="demo",
        )
        pool_b, holdout_b = build_fixed_validation_examples(
            codes=self.codes,
            gaps=self.gaps,
            seq_len=32,
            mask_len_range=(8, 12),
            mask_token=99,
            activity_per_frame=self.activity,
            activity_low_thr=0.01,
            activity_high_thr=0.80,
            examples_per_band=6,
            mask_stride=1,
            seed=123,
            sample_name="demo",
        )

        self.assertEqual(len(pool_a["high_activity"]), 6)
        self.assertEqual(len(pool_a["low_activity"]), 6)
        self.assertEqual(
            [(ex.window_start, ex.mask_start, ex.mask_len) for ex in pool_a["high_activity"]],
            [(ex.window_start, ex.mask_start, ex.mask_len) for ex in pool_b["high_activity"]],
        )
        self.assertEqual(
            [(ex.window_start, ex.mask_start, ex.mask_len) for ex in pool_a["low_activity"]],
            [(ex.window_start, ex.mask_start, ex.mask_len) for ex in pool_b["low_activity"]],
        )
        self.assertEqual(holdout_a, holdout_b)

        for ex in pool_a["high_activity"]:
            self.assertGreaterEqual(ex.mask_mean_activity, 0.80)
            self.assertTrue(is_non_gap_window(ex.window_start, 32, self.gaps))

        for ex in pool_a["low_activity"]:
            self.assertLessEqual(ex.mask_mean_activity, 0.01)
            self.assertTrue(is_non_gap_window(ex.window_start, 32, self.gaps))

        for start, _ in holdout_a:
            self.assertFalse(is_non_gap_window(start, 32, self.gaps, holdout_a))

        blocked_subset = holdout_a[:1]
        train_ds = ActivityAwareMaskedSpanDataset(
            codes=self.codes,
            gaps=self.gaps,
            seq_len=16,
            mask_len_range=(4, 6),
            mask_token=99,
            virtual_size=32,
            activity_per_frame=self.activity,
            token_change_per_frame=self.activity,
            activity_low_thr=0.01,
            activity_high_thr=0.80,
            weighted_sampling=False,
            dead_window_min_mean=0.01,
            dead_window_min_ratio=0.03,
            blocked_ranges=blocked_subset,
            mask_stride=1,
            activity_guided_masking=False,
        )
        for start in train_ds.starts:
            self.assertTrue(is_non_gap_window(start, 16, self.gaps, blocked_subset))

    def test_low_band_keeps_quiet_spans_that_training_filter_would_drop(self):
        pool, _ = build_fixed_validation_examples(
            codes=self.codes,
            gaps=self.gaps,
            seq_len=32,
            mask_len_range=(8, 12),
            mask_token=99,
            activity_per_frame=self.activity,
            activity_low_thr=0.01,
            activity_high_thr=0.80,
            examples_per_band=4,
            mask_stride=1,
            seed=7,
            sample_name="demo",
        )
        self.assertTrue(all(ex.mask_mean_activity <= 0.01 for ex in pool["low_activity"]))

    def test_raises_if_one_band_has_no_candidates(self):
        with self.assertRaises(ValueError):
            build_fixed_validation_examples(
                codes=self.codes,
                gaps=self.gaps,
                seq_len=32,
                mask_len_range=(8, 12),
                mask_token=99,
                activity_per_frame=np.ones(160, dtype=np.float32),
                activity_low_thr=0.01,
                activity_high_thr=0.80,
                examples_per_band=4,
                mask_stride=1,
                seed=7,
                sample_name="demo",
            )


if __name__ == "__main__":
    unittest.main()
