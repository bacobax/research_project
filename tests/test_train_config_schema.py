import sys
import unittest
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.config import TrainConfig as SharedTrainConfig
from audio_infill.config import parse_args as parse_config_args
from audio_infill.train import TrainConfig as RuntimeTrainConfig
from audio_infill.train import parse_args as parse_train_args


class TestTrainConfigSchema(unittest.TestCase):
    def test_shared_and_runtime_train_config_fields_match(self):
        shared_fields = [(f.name, f.default) for f in fields(SharedTrainConfig)]
        runtime_fields = [(f.name, f.default) for f in fields(RuntimeTrainConfig)]
        self.assertEqual(shared_fields, runtime_fields)

    def test_invalid_train_config_is_rejected_in_both_parsers(self):
        bad_args = ["--config", "configs/train/base.yaml", "--ctx-left", "10"]
        with self.assertRaises(ValueError):
            parse_config_args(bad_args)
        with self.assertRaises(ValueError):
            parse_train_args(bad_args)

    def test_phase6_options_default_to_previous_behaviour_and_are_validated(self):
        cfg = RuntimeTrainConfig()
        self.assertEqual(cfg.zeroed_context_prob, 0.0)
        self.assertEqual(cfg.codebook_conditioning, "none")
        self.assertTrue(cfg.validation_autocast)
        self.assertIsNone(cfg.audio_selection_metric)
        self.assertEqual(cfg.inpaint_output_mode, "decode_all")
        self.assertTrue(cfg.tb_log_figures)
        from audio_infill.config import validate_train_config

        for key, bad in [
            ("zeroed_context_prob", 1.5),
            ("codebook_conditioning", "bogus"),
            ("depth_layers", 0),
            ("audio_selection_metric", "nll"),
            ("inpaint_output_mode", "bogus"),
        ]:
            bad_cfg = SharedTrainConfig()
            setattr(bad_cfg, key, bad)
            with self.assertRaises(ValueError, msg=key):
                validate_train_config(bad_cfg)


if __name__ == "__main__":
    unittest.main()
