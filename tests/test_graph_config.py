import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.graph import parse_args


class TestGraphConfig(unittest.TestCase):
    def test_can_source_model_shape_from_train_config(self):
        cfg = parse_args(["--config", "configs/train/base.yaml", "--batch-size", "1", "--time-steps", "16"])
        self.assertEqual(cfg.d_model, 512)
        self.assertEqual(cfg.n_heads, 8)
        self.assertEqual(cfg.n_layers, 8)
        self.assertEqual(cfg.max_len, 2048)
        self.assertEqual(cfg.dropout, 0.1)
        self.assertEqual(cfg.batch_size, 1)
        self.assertEqual(cfg.time_steps, 16)

    def test_invalid_graph_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_args(["--codebooks", "0"])


if __name__ == "__main__":
    unittest.main()
