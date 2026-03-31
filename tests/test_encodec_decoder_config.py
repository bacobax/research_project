import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.encodec_decoder_config import parse_args


class TestEncodecDecoderConfig(unittest.TestCase):
    def test_yaml_config_parses(self):
        cfg, _ = parse_args(["--config", "configs/train/encodec_decoder_finetune.yaml"])
        self.assertEqual(cfg.output_dir, "outputs/runs/encodec_decoder")
        self.assertEqual(cfg.run_name, "wav_test_song_decoder_finetune_important")
        self.assertEqual(cfg.encodec_model, "encodec_24khz")
        self.assertEqual(
            cfg.wav_path,
            "data/processed/multigap/wav_test_multigap_4x_1p0s_2p0s_5p0s_10p0s.wav",
        )
        self.assertEqual(cfg.betas, (0.9, 0.95))
        self.assertEqual(cfg.total_steps, 60000)
        self.assertEqual(cfg.validation_save_audio_every, 7500)
        self.assertEqual(cfg.patience, 6)
        self.assertEqual(cfg.stft_n_ffts, (512, 1024, 2048))

    def test_cli_override_works(self):
        cfg, _ = parse_args(
            [
                "--config",
                "configs/train/encodec_decoder_finetune_smoke.yaml",
                "--total-steps",
                "12",
                "--validation-every",
                "0",
                "--validation-save-audio-every",
                "40",
                "--patience",
                "3",
                "--decoder-export-path",
                "outputs/custom_decoder.pt",
            ]
        )
        self.assertEqual(cfg.total_steps, 12)
        self.assertEqual(cfg.validation_every, 0)
        self.assertEqual(cfg.validation_save_audio_every, 40)
        self.assertEqual(cfg.patience, 3)
        self.assertEqual(cfg.decoder_export_path, "outputs/custom_decoder.pt")


if __name__ == "__main__":
    unittest.main()
