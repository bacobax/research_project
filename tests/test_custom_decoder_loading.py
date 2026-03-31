import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import AudioEncoder


class FakeEncodecModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Conv1d(4, 1, kernel_size=1, bias=False)
        self.quantizer = types.SimpleNamespace(bins=1024)
        self.frame_rate = 75
        self.bandwidth = None

    def set_target_bandwidth(self, bandwidth: float) -> None:
        self.bandwidth = bandwidth


class TestCustomDecoderLoading(unittest.TestCase):
    def test_default_audio_encoder_does_not_load_custom_decoder(self):
        fake_model = FakeEncodecModel()
        with mock.patch("audio_infill.train.build_encodec_model", return_value=fake_model), mock.patch(
            "audio_infill.train.load_exported_decoder"
        ) as load_decoder:
            encoder = AudioEncoder(6.0, torch.device("cpu"))
        self.assertIs(encoder.model, fake_model)
        load_decoder.assert_not_called()

    def test_custom_decoder_checkpoint_loads_into_audio_encoder(self):
        fake_model = FakeEncodecModel()

        def fake_load(path, *, decoder, expected_encodec_model, expected_bandwidth):
            with torch.no_grad():
                decoder.weight.fill_(2.0)
            return {"path": path, "encodec_model": expected_encodec_model, "bandwidth": expected_bandwidth}

        with mock.patch("audio_infill.train.build_encodec_model", return_value=fake_model), mock.patch(
            "audio_infill.train.load_exported_decoder",
            side_effect=fake_load,
        ) as load_decoder:
            encoder = AudioEncoder(
                6.0,
                torch.device("cpu"),
                encodec_model="encodec_24khz",
                custom_decoder_checkpoint="outputs/runs/encodec_decoder/demo/artifacts/decoder.pt",
            )
        load_decoder.assert_called_once()
        self.assertTrue(torch.allclose(encoder.model.decoder.weight, torch.full_like(encoder.model.decoder.weight, 2.0)))


if __name__ == "__main__":
    unittest.main()
