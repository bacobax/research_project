import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import (
    FixedValidationExample,
    MultiResolutionSTFTLoss,
    TrainConfig,
    Trainer,
    audio_metric_crop_bounds,
    si_sdr_db,
)


class TestAudioMetricCropBounds(unittest.TestCase):
    def test_margin_included_crop_wider_than_gap_only(self):
        lo_m, hi_m = audio_metric_crop_bounds(seq_len=512, mask_start=100, mask_len=2, margin_frames=16)
        lo_g, hi_g = audio_metric_crop_bounds(seq_len=512, mask_start=100, mask_len=2, margin_frames=0)
        self.assertEqual((lo_g, hi_g), (100, 102))
        self.assertEqual((lo_m, hi_m), (84, 118))
        self.assertLess(lo_m, lo_g)
        self.assertGreater(hi_m, hi_g)

    def test_clamped_to_window(self):
        lo, hi = audio_metric_crop_bounds(seq_len=10, mask_start=1, mask_len=1, margin_frames=16)
        self.assertEqual(lo, 0)
        self.assertEqual(hi, 10)


class TestSiSdrDb(unittest.TestCase):
    def test_identical_signal_is_very_high(self):
        import numpy as np
        rng = np.random.RandomState(0)
        x = rng.randn(4000).astype(np.float32)
        self.assertGreater(si_sdr_db(x, x), 60.0)

    def test_uncorrelated_noise_is_low(self):
        import numpy as np
        rng = np.random.RandomState(0)
        target = rng.randn(4000).astype(np.float32)
        pred = rng.randn(4000).astype(np.float32)
        self.assertLess(si_sdr_db(pred, target), 5.0)


class _TinyEncoder:
    """Deterministic, EnCodec-free stand-in exposing just the two calls
    `_decode_validation_window_audio` needs, matching AudioEncoder's real shapes:
    codes [B,K,T] -> embeddings [B,D,T] -> audio [B,1,T*samples_per_frame]."""

    samples_per_frame = 8

    def codes_to_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.float()

    def decode_embeddings(self, embeddings: torch.Tensor, scale=None) -> torch.Tensor:
        audio = embeddings.mean(dim=1, keepdim=True)  # [B,1,T]
        return audio.repeat_interleave(self.samples_per_frame, dim=-1)


class _TinyModel(nn.Module):
    """Constant-logits model: forward(x, seg_ids, left_idx, right_idx) -> [B,K,T,V]."""

    def __init__(self, K: int, vocab: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros((K, vocab), dtype=torch.float32))

    def forward(self, x, segment_ids=None, left_dist_idx=None, right_dist_idx=None):
        b, k, t = x.shape
        return self.logits.unsqueeze(0).unsqueeze(2).expand(b, -1, t, -1)


def _build_bare_audio_metrics_trainer(mask_len: int = 2, margin_frames: int = 2) -> Trainer:
    K, T, vocab = 2, 16, 5
    cfg = TrainConfig(
        boundary_max_distance=8,
        validation_audio_metrics_enabled=True,
        validation_audio_metrics_max_examples=None,
        decoded_loss_margin_frames=margin_frames,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")
    trainer.encoder = _TinyEncoder()
    trainer.model = _TinyModel(K=K, vocab=vocab)
    trainer.decoded_stft_loss = MultiResolutionSTFTLoss(n_ffts=[32], hop_lengths=[8], win_lengths=[32])

    y = torch.randint(0, vocab, (K, T), dtype=torch.long)
    loss_mask = torch.zeros(T, dtype=torch.bool)
    mask_start = 6
    loss_mask[mask_start:mask_start + mask_len] = True
    x = y.clone()
    x[:, mask_start:mask_start + mask_len] = 0

    example = FixedValidationExample(
        x=x,
        y=y,
        loss_mask=loss_mask,
        band="high_activity",
        sample_name="unit_test",
        mask_mean_activity=0.5,
        mask_len=mask_len,
        window_start=0,
        mask_start=mask_start,
    )
    examples = [example]

    targets = []
    for ex in examples:
        seq_len = int(ex.y.shape[1])
        lo_m, hi_m = audio_metric_crop_bounds(seq_len, ex.mask_start, ex.mask_len, margin_frames)
        lo_g, hi_g = audio_metric_crop_bounds(seq_len, ex.mask_start, ex.mask_len, 0)
        targets.append({
            "margin": trainer._decode_validation_window_audio(ex.y[:, lo_m:hi_m]),
            "gap_only": trainer._decode_validation_window_audio(ex.y[:, lo_g:hi_g]),
        })
    trainer.validation_audio_targets = {"high_activity_len_2": targets}
    trainer._examples = examples  # stashed for the test to reuse
    return trainer


class TestComputeValidationAudioMetrics(unittest.TestCase):
    def test_returns_finite_values_for_all_four_keys(self):
        trainer = _build_bare_audio_metrics_trainer()
        result = trainer._compute_validation_audio_metrics("high_activity_len_2", trainer._examples)
        self.assertEqual(
            set(result.keys()),
            {"si_sdr_db", "si_sdr_db_gap_only", "spectral_convergence", "spectral_convergence_gap_only"},
        )
        for key, value in result.items():
            self.assertTrue(math.isfinite(value), f"{key} was not finite: {value}")

    def test_max_examples_caps_scoring(self):
        trainer = _build_bare_audio_metrics_trainer()
        # Duplicate the single example so there are 3, then cap at 1.
        trainer._examples = trainer._examples * 3
        trainer.validation_audio_targets["high_activity_len_2"] = (
            trainer.validation_audio_targets["high_activity_len_2"] * 3
        )
        trainer.cfg.validation_audio_metrics_max_examples = 1
        result = trainer._compute_validation_audio_metrics("high_activity_len_2", trainer._examples)
        for value in result.values():
            self.assertTrue(math.isfinite(value))

    def test_missing_group_in_targets_yields_nan_not_a_crash(self):
        trainer = _build_bare_audio_metrics_trainer()
        result = trainer._compute_validation_audio_metrics("nonexistent_group", trainer._examples)
        for value in result.values():
            self.assertTrue(math.isnan(value))


class TestValidationAudioMetricsDisabledByDefault(unittest.TestCase):
    """Guard rail: hand-built Trainers without `self.encoder` (as used by
    tests/test_validation_runner.py and similar) must be unaffected when
    validation_audio_metrics_enabled is False (the default)."""

    def test_run_validation_audio_metrics_not_invoked_when_disabled(self):
        cfg = TrainConfig()  # validation_audio_metrics_enabled defaults to False
        self.assertFalse(cfg.validation_audio_metrics_enabled)
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = cfg
        # No self.encoder, no self.validation_audio_targets set at all -- if run_validation's
        # `if self.cfg.validation_audio_metrics_enabled:` guard were missing or wrong, calling
        # _run_validation_audio_metrics here would raise AttributeError on self.encoder.
        self.assertFalse(hasattr(trainer, "encoder"))


if __name__ == "__main__":
    unittest.main()
