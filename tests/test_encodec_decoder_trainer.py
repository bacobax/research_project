import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.encodec_decoder_config import EncodecDecoderTrainConfig
from audio_infill.train_encodec_decoder import (
    DecoderFinetuneTrainer,
    DecoderWindowExample,
    load_gap_sample_ranges,
    make_examples,
)


class FakeEncodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Conv1d(4, 1, kernel_size=1, bias=False)
        self.quantizer = types.SimpleNamespace(bins=1024)
        self.frame_rate = 75
        self.bandwidth = None

    def set_target_bandwidth(self, bandwidth: float) -> None:
        self.bandwidth = bandwidth


class TestEncodecDecoderTrainer(unittest.TestCase):
    def test_load_gap_sample_ranges_from_matching_annotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "song.wav"
            json_path = Path(tmpdir) / "song.json"
            wav_path.write_bytes(b"")
            json_path.write_text(
                '{"gaps":[{"gap_start_sample":10,"gap_end_sample":20},{"gap_start_sample":30,"gap_end_sample":40}]}',
                encoding="utf-8",
            )
            self.assertEqual(load_gap_sample_ranges(str(wav_path)), [(10, 20), (30, 40)])

    def test_make_examples_excludes_gap_overlapping_windows(self):
        class FakeLayer:
            def decode(self, codes):
                return codes.unsqueeze(1).to(torch.float32)

        class TinyModel:
            def __init__(self):
                self.quantizer = types.SimpleNamespace(vq=types.SimpleNamespace(layers=[FakeLayer(), FakeLayer()]))

            def encode(self, x):
                codes = torch.zeros((1, 2, 4), dtype=torch.long, device=x.device)
                scale = torch.ones((1,), dtype=torch.float32, device=x.device)
                return [(codes, scale)]

        wav = torch.zeros((1, 40), dtype=torch.float32)
        examples = make_examples(
            wav,
            model=TinyModel(),
            device=torch.device("cpu"),
            window_samples=10,
            hop_samples=10,
            blocked_sample_ranges=[(10, 20)],
        )
        self.assertEqual([ex.start_sample for ex in examples], [0, 20, 30])

    def test_validation_audio_save_schedule(self):
        trainer = DecoderFinetuneTrainer.__new__(DecoderFinetuneTrainer)
        trainer.cfg = EncodecDecoderTrainConfig(validation_save_audio_every=10)
        self.assertFalse(trainer._should_save_validation_audio(5))
        self.assertTrue(trainer._should_save_validation_audio(10))

    def test_validation_patience_resets_on_improvement(self):
        trainer = DecoderFinetuneTrainer.__new__(DecoderFinetuneTrainer)
        trainer.cfg = EncodecDecoderTrainConfig()
        trainer.validation_enabled = True
        trainer.val_loader = [
            (
                torch.randn(2, 4, 8),
                torch.randn(2, 8),
                torch.ones(2, 1),
                torch.tensor([0, 8]),
            )
        ]
        trainer.device = torch.device("cpu")
        trainer.writer = mock.Mock()
        trainer.encodec = types.SimpleNamespace(decoder=mock.Mock())
        trainer.encodec.decoder.eval = mock.Mock()
        trainer.encodec.decoder.train = mock.Mock()
        trainer._should_save_validation_audio = mock.Mock(return_value=False)
        trainer._compute_loss = mock.Mock(
            return_value=(
                torch.tensor(0.5),
                {
                    "loss": 0.5,
                    "waveform_l1": 0.1,
                    "waveform_l2": 0.0,
                    "stft": 0.4,
                    "stft_spectral_convergence": 0.2,
                    "stft_log_magnitude": 0.2,
                },
                torch.randn(2, 8),
            )
        )
        trainer.best_val_loss = 1.0
        trainer.validation_checks_since_improvement = 4
        trainer.save_checkpoint = mock.Mock()
        trainer.export_decoder = mock.Mock()

        result = trainer.run_validation(step=12)

        self.assertIsNotNone(result)
        avg, improved = result
        self.assertTrue(improved)
        self.assertEqual(avg["loss"], 0.5)
        self.assertEqual(trainer.best_val_loss, 0.5)
        self.assertEqual(trainer.validation_checks_since_improvement, 0)
        trainer.save_checkpoint.assert_called_once_with("best_val")
        trainer.export_decoder.assert_called_once_with("best")

    def test_validation_patience_increments_without_improvement(self):
        trainer = DecoderFinetuneTrainer.__new__(DecoderFinetuneTrainer)
        trainer.cfg = EncodecDecoderTrainConfig()
        trainer.validation_enabled = True
        trainer.val_loader = [
            (
                torch.randn(1, 4, 8),
                torch.randn(1, 8),
                torch.ones(1, 1),
                torch.tensor([0]),
            )
        ]
        trainer.device = torch.device("cpu")
        trainer.writer = mock.Mock()
        trainer.encodec = types.SimpleNamespace(decoder=mock.Mock())
        trainer.encodec.decoder.eval = mock.Mock()
        trainer.encodec.decoder.train = mock.Mock()
        trainer._should_save_validation_audio = mock.Mock(return_value=False)
        trainer._compute_loss = mock.Mock(
            return_value=(
                torch.tensor(1.25),
                {
                    "loss": 1.25,
                    "waveform_l1": 0.5,
                    "waveform_l2": 0.0,
                    "stft": 0.75,
                    "stft_spectral_convergence": 0.3,
                    "stft_log_magnitude": 0.45,
                },
                torch.randn(1, 8),
            )
        )
        trainer.best_val_loss = 0.5
        trainer.validation_checks_since_improvement = 2
        trainer.save_checkpoint = mock.Mock()
        trainer.export_decoder = mock.Mock()

        result = trainer.run_validation(step=24)

        self.assertIsNotNone(result)
        avg, improved = result
        self.assertFalse(improved)
        self.assertEqual(avg["loss"], 1.25)
        self.assertEqual(trainer.best_val_loss, 0.5)
        self.assertEqual(trainer.validation_checks_since_improvement, 3)
        trainer.save_checkpoint.assert_not_called()
        trainer.export_decoder.assert_not_called()

    def test_align_audio_pair_crops_to_common_length(self):
        trainer = DecoderFinetuneTrainer.__new__(DecoderFinetuneTrainer)
        trainer._logged_length_mismatch = False
        pred = torch.randn(2, 36160)
        target = torch.randn(2, 36000)
        aligned_pred, aligned_target = trainer._align_audio_pair(pred, target)
        self.assertEqual(aligned_pred.shape[-1], 36000)
        self.assertEqual(aligned_target.shape[-1], 36000)

    def test_checkpoint_resume_restores_step_and_best_metrics(self):
        examples = [
            DecoderWindowExample(
                embeddings=torch.randn(4, 8),
                target_audio=torch.randn(8),
                scale=torch.ones(1),
                start_sample=0,
            ),
            DecoderWindowExample(
                embeddings=torch.randn(4, 8),
                target_audio=torch.randn(8),
                scale=torch.ones(1),
                start_sample=8,
            ),
        ]
        fake_wav = torch.randn(1, 16)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EncodecDecoderTrainConfig(
                output_dir=tmpdir,
                run_name="decoder_resume",
                total_steps=2,
                validation_every=0,
            )
            with mock.patch("audio_infill.train_encodec_decoder.build_encodec_model", return_value=FakeEncodec()), mock.patch(
                "audio_infill.train_encodec_decoder.load_audio_mono",
                return_value=fake_wav,
            ), mock.patch("audio_infill.train_encodec_decoder.make_examples", return_value=examples):
                trainer = DecoderFinetuneTrainer(cfg)
                trainer.global_step = 3
                trainer.best_val_loss = 0.25
                trainer.best_train_loss = 0.5
                trainer.validation_checks_since_improvement = 2
                trainer.save_checkpoint("resume_case")

            with mock.patch("audio_infill.train_encodec_decoder.build_encodec_model", return_value=FakeEncodec()), mock.patch(
                "audio_infill.train_encodec_decoder.load_audio_mono",
                return_value=fake_wav,
            ), mock.patch("audio_infill.train_encodec_decoder.make_examples", return_value=examples):
                reloaded = DecoderFinetuneTrainer(cfg)
                reloaded.load_checkpoint(str(cfg.checkpoint_dir / "resume_case.pt"))

            self.assertEqual(reloaded.global_step, 3)
            self.assertEqual(reloaded.best_val_loss, 0.25)
            self.assertEqual(reloaded.best_train_loss, 0.5)
            self.assertEqual(reloaded.validation_checks_since_improvement, 2)


if __name__ == "__main__":
    unittest.main()
