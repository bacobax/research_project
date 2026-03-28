import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.config import parse_args


class TestConfigParsing(unittest.TestCase):
    def test_loads_explicit_yaml_config(self):
        cfg, _ = parse_args(["--config", "configs/train/base.yaml"])
        self.assertEqual(cfg.output_dir, "outputs/runs")
        self.assertEqual(cfg.run_name, "infiller")
        self.assertEqual(cfg.betas, (0.9, 0.95))
        self.assertEqual(cfg.device, "auto")
        self.assertEqual(cfg.boundary_max_distance, 128)
        self.assertEqual(cfg.activity_smooth_kernel, 9)
        self.assertAlmostEqual(cfg.activity_low_quantile, 0.30)
        self.assertAlmostEqual(cfg.activity_high_quantile, 0.70)
        self.assertTrue(cfg.weighted_sampling)
        self.assertTrue(cfg.activity_guided_masking)

    def test_cli_override_still_works(self):
        cfg, _ = parse_args([
            "--config",
            "configs/train/longrun.yaml",
            "--total-steps",
            "123",
            "--device",
            "cpu",
            "--boundary-max-distance",
            "64",
            "--no-weighted-sampling",
            "--no-activity-guided-masking",
        ])
        self.assertEqual(cfg.total_steps, 123)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.boundary_max_distance, 64)
        self.assertFalse(cfg.weighted_sampling)
        self.assertFalse(cfg.activity_guided_masking)


if __name__ == "__main__":
    unittest.main()
