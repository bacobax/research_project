import json
import tempfile
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
        self.assertFalse(cfg.use_encoder_decoder)

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
            "--use-encoder-decoder",
        ])
        self.assertEqual(cfg.total_steps, 123)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.boundary_max_distance, 64)
        self.assertFalse(cfg.weighted_sampling)
        self.assertFalse(cfg.activity_guided_masking)
        self.assertTrue(cfg.use_encoder_decoder)

    def test_loads_nuvole_bianche_short_gap_train_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample = "short_gap_fixture"
            sample_dir = Path(tmpdir)
            (sample_dir / f"{sample}.wav").touch()
            (sample_dir / f"{sample}.json").write_text(
                json.dumps(
                    {
                        "sr": 24000,
                        "gaps": [
                            {"gap_start_s": 1.0, "gap_end_s": 1.015},
                            {"gap_start_s": 2.0, "gap_end_s": 2.020},
                            {"gap_start_s": 3.0, "gap_end_s": 3.030},
                            {"gap_start_s": 4.0, "gap_end_s": 4.050},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cfg, _ = parse_args(
                [
                    "--config",
                    "configs/train/nuvole_bianche_short_gaps_encoder_decoder.yaml",
                    "--ds-dir",
                    tmpdir,
                    "--sample",
                    sample,
                ]
            )

        self.assertFalse(cfg.auto_hparam)
        self.assertFalse(cfg.curriculum)
        self.assertTrue(cfg.use_encoder_decoder)
        self.assertTrue(cfg.decoded_loss_enabled)
        self.assertEqual(cfg.seq_len, 512)
        self.assertEqual(cfg.max_len, 512)
        self.assertEqual((cfg.mask_len_min, cfg.mask_len_max), (1, 4))
        self.assertEqual(cfg.validation_mask_lengths, (1, 2, 3, 4))
        self.assertEqual(cfg.device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
