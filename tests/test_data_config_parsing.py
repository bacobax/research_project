import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.make_gapped_dataset import parse_args


class TestDataConfigParsing(unittest.TestCase):
    def test_loads_data_yaml_config(self):
        cfg = parse_args(["--config", "configs/data/single_gap.yaml"])
        self.assertEqual(cfg.wav, "data/raw/wav_test.wav")
        self.assertEqual(cfg.outdir, "data/processed/single_gap")
        self.assertEqual(cfg.gap_seconds, [0.5, 1.0, 2.0, 5.0, 10.0])
        self.assertEqual(cfg.num_gaps, 1)
        self.assertFalse(cfg.prefer_pow2)

    def test_cli_override_still_works(self):
        cfg = parse_args(
            [
                "--config",
                "configs/data/multigap.yaml",
                "--num-gaps",
                "2",
                "--gap-seconds",
                "3.0",
                "4.0",
                "--center-mode",
                "middle",
            ]
        )
        self.assertEqual(cfg.num_gaps, 2)
        self.assertEqual(cfg.gap_seconds, [3.0, 4.0])
        self.assertEqual(cfg.center_mode, "middle")

    def test_invalid_gap_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_args(["--config", "configs/data/single_gap.yaml", "--gap-seconds", "-1.0"])


if __name__ == "__main__":
    unittest.main()
