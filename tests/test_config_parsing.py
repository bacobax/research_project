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
        self.assertEqual(cfg.encodec_model, "encodec_24khz")
        self.assertIsNone(cfg.custom_decoder_checkpoint)
        self.assertTrue(cfg.weighted_sampling)
        self.assertTrue(cfg.activity_guided_masking)
        self.assertFalse(cfg.use_encoder_decoder)
        self.assertFalse(cfg.use_retrieval_conditioning)
        self.assertEqual(cfg.retrieval_feature_type, "mel")
        self.assertEqual(cfg.retrieval_pool_mode, "mean_std")
        self.assertEqual(cfg.retrieval_bank_mode, "bucketed")
        self.assertEqual(cfg.retrieval_bucket_lengths, ())
        self.assertTrue(cfg.retrieval_prebuild_banks)

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
            "--encodec-model",
            "encodec_24khz",
            "--custom-decoder-checkpoint",
            "outputs/runs/encodec_decoder/demo/artifacts/decoder.pt",
            "--no-weighted-sampling",
            "--no-activity-guided-masking",
            "--use-encoder-decoder",
            "--use-retrieval-conditioning",
            "--retrieval-feature-type",
            "stft_mag",
            "--retrieval-pool-mode",
            "mean",
            "--retrieval-top-k",
            "3",
            "--retrieval-candidate-stride-frames",
            "12",
            "--retrieval-exclusion-margin-frames",
            "4",
            "--retrieval-bucket-lengths",
            "187",
            "375",
            "750",
        ])
        self.assertEqual(cfg.total_steps, 123)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.boundary_max_distance, 64)
        self.assertEqual(cfg.encodec_model, "encodec_24khz")
        self.assertEqual(cfg.custom_decoder_checkpoint, "outputs/runs/encodec_decoder/demo/artifacts/decoder.pt")
        self.assertFalse(cfg.weighted_sampling)
        self.assertFalse(cfg.activity_guided_masking)
        self.assertTrue(cfg.use_encoder_decoder)
        self.assertTrue(cfg.use_retrieval_conditioning)
        self.assertEqual(cfg.retrieval_feature_type, "stft_mag")
        self.assertEqual(cfg.retrieval_pool_mode, "mean")
        self.assertEqual(cfg.retrieval_top_k, 3)
        self.assertEqual(cfg.retrieval_candidate_stride_frames, 12)
        self.assertEqual(cfg.retrieval_exclusion_margin_frames, 4)
        self.assertEqual(cfg.retrieval_bucket_lengths, (187, 375, 750))


if __name__ == "__main__":
    unittest.main()
