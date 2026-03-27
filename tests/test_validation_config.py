import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.config import parse_args as parse_config_args
from audio_infill.train import parse_args as parse_train_args


class TestValidationConfig(unittest.TestCase):
    def test_defaults_keep_validation_disabled(self):
        cfg, _ = parse_config_args(["--config", "configs/train/base.yaml"])
        self.assertEqual(cfg.validation_every, 0)
        self.assertEqual(cfg.validation_examples_per_band, 64)
        self.assertIsNone(cfg.validation_batch_size)
        self.assertFalse(cfg.validation_inspection_enabled)
        self.assertEqual(cfg.validation_inspection_examples_per_group, 1)
        self.assertIsNone(cfg.validation_crop_context_frames)
        self.assertTrue(cfg.validation_save_artifacts)

    def test_cli_overrides_apply_in_train_parser(self):
        cfg, _ = parse_train_args(
            [
                "--config",
                "configs/train/base.yaml",
                "--validation-every",
                "12",
                "--validation-examples-per-band",
                "8",
                "--validation-batch-size",
                "4",
                "--validation-inspection-enabled",
                "--validation-inspection-examples-per-group",
                "2",
                "--validation-crop-context-frames",
                "120",
                "--no-validation-save-artifacts",
            ]
        )
        self.assertEqual(cfg.validation_every, 12)
        self.assertEqual(cfg.validation_examples_per_band, 8)
        self.assertEqual(cfg.validation_batch_size, 4)
        self.assertTrue(cfg.validation_inspection_enabled)
        self.assertEqual(cfg.validation_inspection_examples_per_group, 2)
        self.assertEqual(cfg.validation_crop_context_frames, 120)
        self.assertFalse(cfg.validation_save_artifacts)


if __name__ == "__main__":
    unittest.main()
