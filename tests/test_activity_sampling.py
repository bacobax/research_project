import unittest
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import ActivityAwareMaskedSpanDataset


class TestActivityAwareSampling(unittest.TestCase):
    def _make_dataset(self, guided: bool, weighted: bool):
        k, f = 4, 240
        codes = torch.randint(low=0, high=16, size=(k, f), dtype=torch.long)
        activity = np.zeros(f, dtype=np.float32)
        activity[40:120] = np.linspace(0.1, 1.0, 80, dtype=np.float32)
        activity[120:180] = np.linspace(1.0, 0.1, 60, dtype=np.float32)
        token_change = np.clip(activity * 0.9, 0.0, 1.0)
        gaps = [(185, 205)]

        return ActivityAwareMaskedSpanDataset(
            codes=codes,
            gaps=gaps,
            seq_len=64,
            mask_len_range=(8, 16),
            mask_token=99,
            virtual_size=128,
            activity_per_frame=activity,
            token_change_per_frame=token_change,
            activity_low_thr=float(np.quantile(activity, 0.3)),
            activity_high_thr=float(np.quantile(activity, 0.7)),
            weighted_sampling=weighted,
            dead_window_min_mean=0.01,
            dead_window_min_ratio=0.03,
            regime_probs={"active": 0.45, "transition": 0.30, "low_activity": 0.15, "uniform": 0.10},
            mask_stride=1,
            activity_guided_masking=guided,
        )

    def test_returns_expected_tuple_shapes(self):
        ds = self._make_dataset(guided=True, weighted=True)
        x, y, loss_mask = ds[0]

        self.assertEqual(tuple(x.shape), (4, 64))
        self.assertEqual(tuple(y.shape), (4, 64))
        self.assertEqual(tuple(loss_mask.shape), (64,))
        self.assertEqual(loss_mask.dtype, torch.bool)
        self.assertGreater(int(loss_mask.sum().item()), 0)

    def test_disabling_guidance_preserves_uniform_masking(self):
        ds = self._make_dataset(guided=False, weighted=False)
        for i in range(8):
            x, y, loss_mask = ds[i]
            self.assertTrue(torch.equal(y[:, ~loss_mask], x[:, ~loss_mask]))
            self.assertTrue(torch.all(x[:, loss_mask] == 99))


if __name__ == "__main__":
    unittest.main()
