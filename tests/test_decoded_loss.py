import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.config import parse_args as parse_config_args
from audio_infill.train import AudioEncoder, MultiResolutionSTFTLoss, TrainConfig, Trainer, parse_args as parse_train_args


class TestDecodedLossConfig(unittest.TestCase):
    def test_new_decoded_loss_config_parses_in_shared_and_runtime_paths(self):
        args = ["--config", "configs/train/multigap_decoded_loss.yaml"]
        shared_cfg, _ = parse_config_args(args)
        runtime_cfg, _ = parse_train_args(args)

        for cfg in [shared_cfg, runtime_cfg]:
            self.assertTrue(cfg.decoded_loss_enabled)
            self.assertAlmostEqual(cfg.decoded_loss_weight, 0.10)
            self.assertEqual(cfg.decoded_loss_start_step, 7000)
            self.assertEqual(cfg.decoded_loss_every, 4)
            self.assertEqual(cfg.decoded_loss_max_items, 1)
            self.assertEqual(cfg.decoded_loss_margin_frames, 16)
            self.assertAlmostEqual(cfg.decoded_loss_temperature, 1.0)
            self.assertAlmostEqual(cfg.decoded_loss_waveform_l1_weight, 0.25)
            self.assertAlmostEqual(cfg.decoded_loss_stft_weight, 1.0)
            self.assertAlmostEqual(cfg.decoded_loss_spectral_convergence_weight, 1.0)
            self.assertAlmostEqual(cfg.decoded_loss_log_magnitude_weight, 1.0)
            self.assertEqual(tuple(cfg.decoded_loss_n_ffts), (512, 1024, 2048))
            self.assertEqual(tuple(cfg.decoded_loss_hop_lengths), (128, 256, 512))
            self.assertEqual(tuple(cfg.decoded_loss_win_lengths), (512, 1024, 2048))


class TestDecodedLossWiring(unittest.TestCase):
    def test_decode_embeddings_temporarily_enables_decoder_training_mode(self):
        class FakeDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.mode_during_forward = None

            def forward(self, x):
                self.mode_during_forward = self.training
                return x

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = FakeDecoder()

        encoder = AudioEncoder.__new__(AudioEncoder)
        encoder.device = torch.device("cpu")
        encoder.model = FakeModel()
        encoder.model.decoder.eval()

        embeddings = torch.randn(1, 4, 8, requires_grad=True)
        decoded = encoder.decode_embeddings(embeddings)
        self.assertTrue(encoder.model.decoder.mode_during_forward)
        self.assertFalse(encoder.model.decoder.training)
        decoded.sum().backward()

    def test_schedule_logic(self):
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = TrainConfig(
            decoded_loss_enabled=True,
            decoded_loss_weight=0.1,
            decoded_loss_start_step=5,
            decoded_loss_every=3,
        )
        trainer.decoded_loss_enabled = True

        self.assertFalse(trainer._should_apply_decoded_loss(4))
        self.assertTrue(trainer._should_apply_decoded_loss(5))
        self.assertFalse(trainer._should_apply_decoded_loss(6))
        self.assertFalse(trainer._should_apply_decoded_loss(7))
        self.assertTrue(trainer._should_apply_decoded_loss(8))

    def test_training_losses_skip_decoded_term_when_disabled(self):
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = TrainConfig(decoded_loss_enabled=False, decoded_loss_weight=0.1)
        trainer.decoded_loss_enabled = False
        trainer._compute_loss = lambda logits, y, loss_mask: torch.tensor(2.0)
        trainer._compute_decoded_domain_loss = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run"))

        total, token, metrics = trainer._compute_training_losses(
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            step=0,
        )
        self.assertAlmostEqual(total.item(), 2.0)
        self.assertAlmostEqual(token.item(), 2.0)
        self.assertEqual(metrics["decoded_loss_total"], 0.0)

    def test_training_losses_include_decoded_term_when_enabled(self):
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = TrainConfig(
            decoded_loss_enabled=True,
            decoded_loss_weight=0.1,
            decoded_loss_start_step=0,
            decoded_loss_every=1,
        )
        trainer.decoded_loss_enabled = True
        trainer._compute_loss = lambda logits, y, loss_mask: torch.tensor(2.0)
        trainer._compute_decoded_domain_loss = lambda logits, y, loss_mask: (
            torch.tensor(0.5),
            {
                "decoded_loss_total": 0.5,
                "decoded_loss_waveform_l1": 0.1,
                "decoded_loss_stft": 0.2,
                "decoded_loss_spectral_convergence": 0.3,
                "decoded_loss_log_magnitude": 0.4,
                "decoded_loss_items": 1.0,
            },
        )

        total, token, metrics = trainer._compute_training_losses(
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            step=0,
        )
        self.assertAlmostEqual(total.item(), 2.5)
        self.assertAlmostEqual(token.item(), 2.0)
        self.assertAlmostEqual(metrics["decoded_loss_total"], 0.5)

    def test_mrstft_loss_is_finite(self):
        loss_fn = MultiResolutionSTFTLoss(
            n_ffts=(128, 256),
            hop_lengths=(32, 64),
            win_lengths=(128, 256),
            spectral_convergence_weight=1.0,
            log_magnitude_weight=1.0,
        )
        pred = torch.randn(2, 4096)
        target = torch.randn(2, 4096)
        out = loss_fn(pred, target)
        self.assertTrue(torch.isfinite(out["total"]))
        self.assertTrue(torch.isfinite(out["spectral_convergence"]))
        self.assertTrue(torch.isfinite(out["log_magnitude"]))


if __name__ == "__main__":
    unittest.main()
