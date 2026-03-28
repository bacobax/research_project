import json
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


class ValidationSpec:
    def __init__(self, band: str, mask_len: int):
        self.band = band
        self.mask_len = mask_len


class DummyWriter:
    def __init__(self):
        self.scalars = []
        self.figures = []
        self.flushed = False

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), int(step)))

    def add_figure(self, name, figure, step):
        self.figures.append((name, int(step)))

    def flush(self):
        self.flushed = True

    def close(self):
        pass


class ConstantZeroModel(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.vocab = vocab

    def forward(self, x: torch.Tensor, segment_ids=None, left_dist_idx=None, right_dist_idx=None) -> torch.Tensor:
        b, k, t = x.shape
        return torch.zeros((b, k, t, self.vocab), dtype=torch.float32, device=x.device) + self.bias


class DummyEncoder:
    def __init__(self):
        self.frame_rate = 75

    def codes_to_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        return codes.float().sum(dim=1, keepdim=True)

    def decode_embeddings(self, embeddings: torch.Tensor, scale=None) -> torch.Tensor:
        return embeddings.float()


def make_example(band: str, sample_name: str, window_start: int) -> FixedValidationExample:
    y = torch.zeros((2, 6), dtype=torch.long)
    x = y.clone()
    x[:, 2:4] = 9
    loss_mask = torch.zeros(6, dtype=torch.bool)
    loss_mask[2:4] = True
    return FixedValidationExample(
        x=x,
        y=y,
        loss_mask=loss_mask,
        band=band,
        sample_name=sample_name,
        mask_mean_activity=0.9 if band == "high_activity" else 0.0,
        mask_len=2,
        window_start=window_start,
        mask_start=2,
    )


class TestValidationInspection(unittest.TestCase):
    def test_validation_inspection_logs_figures_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig(
                output_dir=tmpdir,
                run_name="inspection",
                validation_inspection_enabled=True,
                validation_crop_context_frames=1,
                validation_save_artifacts=True,
            )
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            high_example = make_example("high_activity", "high_demo", 4)
            low_example = make_example("low_activity", "low_demo", 8)

            trainer = Trainer.__new__(Trainer)
            trainer.cfg = cfg
            trainer.device = torch.device("cpu")
            trainer.writer = DummyWriter()
            trainer.model = ConstantZeroModel(vocab=5)
            trainer.encoder = DummyEncoder()
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
            trainer.scaler = torch.amp.GradScaler(enabled=False)
            trainer.validation_enabled = True
            trainer.validation_dataloaders = {
                "high_activity_len_2": DataLoader(FixedMaskedSpanDataset([high_example]), batch_size=1, shuffle=False),
                "low_activity_len_2": DataLoader(FixedMaskedSpanDataset([low_example]), batch_size=1, shuffle=False),
            }
            trainer.validation_group_specs = {
                "high_activity_len_2": ValidationSpec("high_activity", 2),
                "low_activity_len_2": ValidationSpec("low_activity", 2),
            }
            trainer.validation_inspection_examples = {
                "high_activity_len_2": [high_example],
                "low_activity_len_2": [low_example],
            }
            trainer.best_loss = float("inf")
            trainer.best_val_loss = float("inf")
            trainer.global_step = 0

            result = trainer.run_validation(step=5)

            self.assertIsNotNone(result)
            figure_names = {name for name, _ in trainer.writer.figures}
            self.assertIn(
                "validation_inspect/high_activity_len_2/high_demo__high_activity_len_2__idx00__ws000004__ms0002__ml0002/spectrogram_full",
                figure_names,
            )
            self.assertIn(
                "validation_inspect/low_activity_len_2/low_demo__low_activity_len_2__idx00__ws000008__ms0002__ml0002/waveform",
                figure_names,
            )
            self.assertTrue(trainer.writer.flushed)

            step_dir = cfg.samples_dir / "validation" / "step_5"
            manifest_path = step_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["examples"]), 2)

            first_artifact_dir = Path(manifest["examples"][0]["artifact_dir"])
            self.assertTrue((first_artifact_dir / "bundle.pt").exists())
            self.assertTrue((first_artifact_dir / "metadata.json").exists())
            self.assertTrue((first_artifact_dir / "pred_window.wav").exists())
            self.assertTrue((first_artifact_dir / "target_gap_crop.wav").exists())


if __name__ == "__main__":
    unittest.main()
