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


if __name__ == "__main__":
    unittest.main()
