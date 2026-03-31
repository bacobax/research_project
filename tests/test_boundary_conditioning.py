import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import (
    EncoderDecoderCodebookInfiller,
    JointCodebookInfiller,
    TrainConfig,
    Trainer,
    build_boundary_condition_tensors,
)


class DummyWriter:
    def add_scalar(self, name, value, step):
        pass

    def close(self):
        pass


class TestBoundaryConditioning(unittest.TestCase):
    def test_build_boundary_condition_tensors_single_mask(self):
        loss_mask = torch.tensor([False, False, True, True, False, False])
        segment_ids, left_idx, right_idx = build_boundary_condition_tensors(loss_mask, max_distance=2)

        self.assertEqual(segment_ids.shape, (1, 6))
        self.assertTrue(torch.equal(segment_ids[0], torch.tensor([0, 0, 1, 1, 2, 2])))
        self.assertTrue(torch.equal(left_idx[0], torch.tensor([0, 1, 2, 3, 4, 4])))
        self.assertTrue(torch.equal(right_idx[0], torch.tensor([4, 4, 4, 3, 2, 1])))

    def test_build_boundary_condition_tensors_rejects_non_contiguous_mask(self):
        loss_mask = torch.tensor([[False, True, False, True]])
        with self.assertRaises(ValueError):
            build_boundary_condition_tensors(loss_mask, max_distance=4)

    def test_model_explicit_and_fallback_boundary_tensors_match(self):
        model = JointCodebookInfiller(
            K=2,
            bins=8,
            mask_token=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_len=16,
            dropout=0.0,
            boundary_max_distance=4,
        )
        x = torch.randint(0, 8, (1, 2, 6))
        x[:, :, 2:4] = 8
        loss_mask = torch.tensor([[False, False, True, True, False, False]])
        segment_ids, left_idx, right_idx = build_boundary_condition_tensors(loss_mask, max_distance=4)

        logits_explicit = model(x, segment_ids, left_idx, right_idx)
        logits_fallback = model(x)
        self.assertEqual(logits_explicit.shape, (1, 2, 6, 8))
        self.assertTrue(torch.allclose(logits_explicit, logits_fallback, atol=1e-6))

    def test_encoder_decoder_explicit_and_fallback_boundary_tensors_match(self):
        model = EncoderDecoderCodebookInfiller(
            K=2,
            bins=8,
            mask_token=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_len=16,
            dropout=0.0,
            boundary_max_distance=4,
        )
        x = torch.randint(0, 8, (1, 2, 6))
        x[:, :, 2:4] = 8
        loss_mask = torch.tensor([[False, False, True, True, False, False]])
        segment_ids, left_idx, right_idx = build_boundary_condition_tensors(loss_mask, max_distance=4)

        logits_explicit = model(x, segment_ids, left_idx, right_idx)
        logits_fallback = model(x)
        self.assertEqual(logits_explicit.shape, (1, 2, 6, 8))
        self.assertTrue(torch.allclose(logits_explicit, logits_fallback, atol=1e-6))

    def test_encoder_decoder_uses_context_and_gap_lengths(self):
        class CaptureEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_src = None
                self.last_padding = None

            def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
                self.last_src = src
                self.last_padding = src_key_padding_mask
                return src

        class CaptureDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_tgt = None
                self.last_memory = None
                self.last_tgt_padding = None
                self.last_memory_padding = None

            def forward(
                self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_is_causal=False,
                memory_is_causal=False,
            ):
                self.last_tgt = tgt
                self.last_memory = memory
                self.last_tgt_padding = tgt_key_padding_mask
                self.last_memory_padding = memory_key_padding_mask
                return tgt

        model = EncoderDecoderCodebookInfiller(
            K=2,
            bins=8,
            mask_token=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_len=16,
            dropout=0.0,
            boundary_max_distance=4,
        )
        model.enc = CaptureEncoder()
        model.dec = CaptureDecoder()
        x = torch.randint(0, 8, (1, 2, 6))
        x[:, :, 2:4] = 8

        logits = model(x)
        self.assertEqual(logits.shape, (1, 2, 6, 8))
        self.assertEqual(tuple(model.enc.last_src.shape[:2]), (1, 4))
        self.assertEqual(tuple(model.dec.last_tgt.shape[:2]), (1, 2))

    def test_load_checkpoint_without_boundary_embeddings_uses_compat_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig(output_dir=tmpdir, run_name="boundary_ckpt")
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            trainer = Trainer.__new__(Trainer)
            trainer.cfg = cfg
            trainer.device = torch.device("cpu")
            trainer.writer = DummyWriter()
            trainer.model = JointCodebookInfiller(
                K=2,
                bins=8,
                mask_token=8,
                d_model=16,
                n_heads=4,
                n_layers=1,
                max_len=16,
                dropout=0.0,
                boundary_max_distance=cfg.boundary_max_distance,
            )
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
            trainer.scaler = torch.amp.GradScaler(enabled=False)
            trainer.best_loss = 1.0
            trainer.best_val_loss = 0.5
            trainer.global_step = 7

            legacy_state = {
                k: v
                for k, v in trainer.model.state_dict().items()
                if k not in {"segment_emb.weight", "left_distance_emb.weight", "right_distance_emb.weight"}
            }
            ckpt_path = cfg.checkpoint_dir / "legacy.pt"
            torch.save(
                {
                    "step": trainer.global_step,
                    "model": legacy_state,
                    "optimizer": trainer.optimizer.state_dict(),
                    "scaler": trainer.scaler.state_dict(),
                    "best_loss": trainer.best_loss,
                    "best_val_loss": trainer.best_val_loss,
                },
                ckpt_path,
            )

            reloaded = Trainer.__new__(Trainer)
            reloaded.cfg = cfg
            reloaded.device = torch.device("cpu")
            reloaded.writer = DummyWriter()
            reloaded.model = JointCodebookInfiller(
                K=2,
                bins=8,
                mask_token=8,
                d_model=16,
                n_heads=4,
                n_layers=1,
                max_len=16,
                dropout=0.0,
                boundary_max_distance=cfg.boundary_max_distance,
            )
            reloaded.optimizer = torch.optim.AdamW(reloaded.model.parameters(), lr=1e-3)
            reloaded.scaler = torch.amp.GradScaler(enabled=False)
            reloaded.load_checkpoint(str(ckpt_path))

            self.assertEqual(reloaded.global_step, 7)
            self.assertEqual(reloaded.best_val_loss, 0.5)

    def test_load_checkpoint_rejects_architecture_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig(output_dir=tmpdir, run_name="boundary_ckpt")
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            trainer = Trainer.__new__(Trainer)
            trainer.cfg = cfg
            trainer.device = torch.device("cpu")
            trainer.writer = DummyWriter()
            trainer.model = JointCodebookInfiller(
                K=2,
                bins=8,
                mask_token=8,
                d_model=16,
                n_heads=4,
                n_layers=1,
                max_len=16,
                dropout=0.0,
                boundary_max_distance=cfg.boundary_max_distance,
            )
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
            trainer.scaler = torch.amp.GradScaler(enabled=False)
            trainer.best_loss = 1.0
            trainer.best_val_loss = 0.5
            trainer.global_step = 7

            ckpt_path = cfg.checkpoint_dir / "legacy.pt"
            torch.save(
                {
                    "step": trainer.global_step,
                    "model": trainer.model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "scaler": trainer.scaler.state_dict(),
                    "best_loss": trainer.best_loss,
                    "best_val_loss": trainer.best_val_loss,
                    "config": {"use_encoder_decoder": False},
                },
                ckpt_path,
            )

            reloaded = Trainer.__new__(Trainer)
            reloaded.cfg = TrainConfig(output_dir=tmpdir, run_name="boundary_ckpt", use_encoder_decoder=True)
            reloaded.device = torch.device("cpu")
            reloaded.writer = DummyWriter()
            reloaded.model = EncoderDecoderCodebookInfiller(
                K=2,
                bins=8,
                mask_token=8,
                d_model=16,
                n_heads=4,
                n_layers=1,
                max_len=16,
                dropout=0.0,
                boundary_max_distance=cfg.boundary_max_distance,
            )
            reloaded.optimizer = torch.optim.AdamW(reloaded.model.parameters(), lr=1e-3)
            reloaded.scaler = torch.amp.GradScaler(enabled=False)

            with self.assertRaises(ValueError):
                reloaded.load_checkpoint(str(ckpt_path))


if __name__ == "__main__":
    unittest.main()
