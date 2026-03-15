import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import FixedMaskedSpanDataset, FixedValidationExample, TrainConfig, Trainer


class DummyWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), int(step)))

    def close(self):
        pass


class ConstantZeroModel(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.vocab = vocab

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, t = x.shape
        return torch.zeros((b, k, t, self.vocab), dtype=torch.float32, device=x.device) + self.bias


def make_validation_dataset(band: str) -> FixedMaskedSpanDataset:
    examples = []
    for idx in range(2):
        y = torch.zeros((2, 6), dtype=torch.long)
        x = y.clone()
        x[:, 2:4] = 9
        loss_mask = torch.zeros(6, dtype=torch.bool)
        loss_mask[2:4] = True
        examples.append(
            FixedValidationExample(
                x=x,
                y=y,
                loss_mask=loss_mask,
                band=band,
                sample_name=f"{band}_{idx}",
                mask_mean_activity=0.9 if band == "high_activity" else 0.0,
                mask_len=2,
                window_start=idx,
                mask_start=2,
            )
        )
    return FixedMaskedSpanDataset(examples)


class TestValidationRunner(unittest.TestCase):
    def test_validation_logs_metrics_and_saves_best_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig(output_dir=tmpdir, run_name="runner")
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            trainer = Trainer.__new__(Trainer)
            trainer.cfg = cfg
            trainer.device = torch.device("cpu")
            trainer.writer = DummyWriter()
            trainer.model = ConstantZeroModel(vocab=5)
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
            trainer.scaler = torch.amp.GradScaler(enabled=False)
            trainer.validation_enabled = True
            trainer.validation_dataloaders = {
                "high_activity": DataLoader(make_validation_dataset("high_activity"), batch_size=2, shuffle=False),
                "low_activity": DataLoader(make_validation_dataset("low_activity"), batch_size=2, shuffle=False),
            }
            trainer.best_loss = float("inf")
            trainer.best_val_loss = float("inf")
            trainer.global_step = 0

            result = trainer.run_validation(step=3)

            self.assertIsNotNone(result)
            self.assertAlmostEqual(result["combined_loss"], (result["high_loss"] + result["low_loss"]) / 2.0)
            self.assertTrue(trainer.model.training)
            self.assertTrue((cfg.checkpoint_dir / "best_val.pt").exists())

            scalar_names = {name for name, _, _ in trainer.writer.scalars}
            self.assertIn("val/high_loss", scalar_names)
            self.assertIn("val/low_loss", scalar_names)
            self.assertIn("val/combined_loss", scalar_names)

            previous_best = trainer.best_val_loss
            trainer.best_val_loss = 0.0
            trainer.run_validation(step=4)
            self.assertEqual(trainer.best_val_loss, 0.0)
            self.assertLessEqual(previous_best, result["combined_loss"])


if __name__ == "__main__":
    unittest.main()
