import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import (
    ActivityAwareMaskedSpanDataset,
    AudioEncoder,
    FixedValidationExample,
    MultiSongMaskedSpanDataset,
    Trainer,
    TrainConfig,
)

HOP = 320
SR = 24000


def make_trainer(prob: float = 1.0):
    torch.manual_seed(0)
    n = SR * 8
    wav = (0.3 * torch.sin(torch.arange(n) * 0.05) + 0.05 * torch.randn(n)).unsqueeze(0).float()
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = TrainConfig(target_sr=SR, zeroed_context_prob=prob, zeroed_context_warmup_frames=40)
    trainer.device = torch.device("cpu")
    trainer.encoder = AudioEncoder(6.0, torch.device("cpu"))
    trainer.mask_token = trainer.encoder.mask_token
    trainer.wav = wav
    trainer.song_wavs = [wav]
    codes, _ = trainer.encoder.encode(wav)
    return trainer, wav, codes


class TestZeroedContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer, cls.wav, cls.codes = make_trainer()

    def _batch(self, s=60, T=64, m0=30, m1=32):
        x = self.codes[:, s:s + T].clone().unsqueeze(0)
        y = x.clone()
        loss_mask = torch.zeros(1, T, dtype=torch.bool)
        loss_mask[0, m0:m1] = True
        x[0, :, m0:m1] = self.trainer.mask_token
        meta = torch.tensor([[0, s]])
        return x, y, loss_mask, meta

    def test_context_changes_only_after_gap_and_targets_untouched(self):
        x, y, lm, meta = self._batch()
        y_before = y.clone()
        out = self.trainer._apply_zeroed_context(x, lm, meta)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.equal(y, y_before))
        # mask tokens are in place
        self.assertTrue((out[0, :, 30:32] == self.trainer.mask_token).all())
        # causal encoder: tokens before the gap match the clean encoding (up to warm-up/boundary
        # effects, which are small)
        before = (out[0, :, :30] == self.codes[:, 60:90]).float().mean().item()
        after = (out[0, :, 32:] == self.codes[:, 92:124]).float().mean().item()
        self.assertEqual(before, 1.0)  # clean before the first touched frame (causal encoder)
        self.assertLess(after, before)

    def test_probability_zero_leaves_batch_identical(self):
        trainer, _, _ = self.trainer, None, None
        trainer.cfg.zeroed_context_prob = 0.0
        try:
            x, y, lm, meta = self._batch()
            self.assertTrue(torch.equal(trainer._apply_zeroed_context(x, lm, meta), x))
        finally:
            trainer.cfg.zeroed_context_prob = 1.0

    def test_validation_example_is_deterministic(self):
        T = 64
        y = self.codes[:, 60:60 + T].clone()
        lm = torch.zeros(T, dtype=torch.bool)
        lm[30:32] = True
        x = y.clone()
        x[:, lm] = self.trainer.mask_token
        ex = FixedValidationExample(x=x, y=y, loss_mask=lm, band="high_activity", sample_name="t",
                                    mask_mean_activity=0.5, mask_len=2, window_start=60, mask_start=30)
        a = self.trainer._zero_context_example(ex)
        b = self.trainer._zero_context_example(ex)
        self.assertTrue(torch.equal(a.x, b.x))
        self.assertTrue(torch.equal(a.y, y))
        self.assertTrue(torch.equal(a.loss_mask, lm))


class TestDatasetMeta(unittest.TestCase):
    def _ds(self):
        torch.manual_seed(0)
        codes = torch.randint(0, 64, (2, 600))
        act = np.ones(600, dtype=np.float32) * 0.5
        return ActivityAwareMaskedSpanDataset(
            codes=codes, gaps=[], seq_len=64, mask_len_range=(1, 4), mask_token=64,
            activity_per_frame=act, token_change_per_frame=act, activity_low_thr=0.1,
            activity_high_thr=0.9, weighted_sampling=False,
        )

    def test_default_returns_three_items(self):
        self.assertEqual(len(self._ds()[0]), 3)

    def test_meta_when_enabled(self):
        multi = MultiSongMaskedSpanDataset([self._ds(), self._ds()], np.ones(2), virtual_size=10)
        multi.enable_meta()
        item = multi[0]
        self.assertEqual(len(item), 4)
        self.assertIn(int(item[3][0]), (0, 1))
        s = int(item[3][1])
        self.assertTrue(0 <= s <= 600 - 64)


if __name__ == "__main__":
    unittest.main()
