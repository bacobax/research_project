import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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


class TinyTrainDataset(Dataset):
    def __init__(self):
        self.recent_metrics = deque()

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        x = torch.zeros((2, 6), dtype=torch.long)
        y = torch.zeros((2, 6), dtype=torch.long)
        loss_mask = torch.zeros(6, dtype=torch.bool)
        loss_mask[1:3] = True
        return x, y, loss_mask

    def pop_recent_metrics(self, max_items=None):
        return []


class TinyModel(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros((2, vocab), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, t = x.shape
        return self.logits.unsqueeze(0).unsqueeze(2).expand(b, -1, t, -1)


def make_validation_loader(band: str) -> DataLoader:
    y = torch.zeros((2, 6), dtype=torch.long)
    x = y.clone()
    x[:, 2:4] = 9
    loss_mask = torch.zeros(6, dtype=torch.bool)
    loss_mask[2:4] = True
    dataset = FixedMaskedSpanDataset(
        [
            FixedValidationExample(
                x=x,
                y=y,
                loss_mask=loss_mask,
                band=band,
                sample_name=band,
                mask_mean_activity=0.9 if band == "high_activity" else 0.0,
                mask_len=2,
                window_start=0,
                mask_start=2,
            )
        ]
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


class TestValidationIntegration(unittest.TestCase):
    def test_train_runs_validation_on_schedule_and_saves_best_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig(
                output_dir=tmpdir,
                run_name="integration",
                total_steps=4,
                warmup_steps=1,
                save_every=100,
                log_every=10,
                test_fill_every=0,
                validation_every=2,
            )
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            trainer = Trainer.__new__(Trainer)
            trainer.cfg = cfg
            trainer.device = torch.device("cpu")
            trainer.writer = DummyWriter()
            trainer.dataset = TinyTrainDataset()
            trainer.dataloader = DataLoader(trainer.dataset, batch_size=1, shuffle=False)
            trainer.validation_enabled = True
            trainer.validation_dataloaders = {
                "high_activity": make_validation_loader("high_activity"),
                "low_activity": make_validation_loader("low_activity"),
            }
            trainer.model = TinyModel(vocab=5)
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
            trainer.scaler = torch.amp.GradScaler(enabled=False)
            trainer.global_step = 0
            trainer.best_loss = float("inf")
            trainer.best_val_loss = float("inf")

            calls = []
            original_run_validation = trainer.run_validation

            def wrapped_run_validation(step):
                calls.append(step)
                return original_run_validation(step)

            trainer.run_validation = wrapped_run_validation
            trainer.train()

            self.assertEqual(calls, [2, 4])
            self.assertTrue((cfg.checkpoint_dir / "best_val.pt").exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig(output_dir=tmpdir, run_name="resume")
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            trainer = Trainer.__new__(Trainer)
            trainer.cfg = cfg
            trainer.device = torch.device("cpu")
            trainer.writer = DummyWriter()
            trainer.model = TinyModel(vocab=5)
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
            trainer.scaler = torch.amp.GradScaler(enabled=False)
            trainer.best_loss = 1.0
            trainer.best_val_loss = 0.25
            trainer.global_step = 3
            trainer.save_checkpoint("resume_case")

            reloaded = Trainer.__new__(Trainer)
            reloaded.cfg = cfg
            reloaded.device = torch.device("cpu")
            reloaded.writer = DummyWriter()
            reloaded.model = TinyModel(vocab=5)
            reloaded.optimizer = torch.optim.AdamW(reloaded.model.parameters(), lr=1e-3)
            reloaded.scaler = torch.amp.GradScaler(enabled=False)
            reloaded.load_checkpoint(str(cfg.checkpoint_dir / "resume_case.pt"))

            self.assertEqual(reloaded.best_val_loss, 0.25)


if __name__ == "__main__":
    unittest.main()
