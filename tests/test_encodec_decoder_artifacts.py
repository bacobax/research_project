import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.encodec_utils import export_decoder_artifact, load_decoder_artifact, load_exported_decoder


class TestEncodecDecoderArtifacts(unittest.TestCase):
    def test_valid_artifact_loads(self):
        decoder = nn.Sequential(nn.Conv1d(8, 8, kernel_size=1))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decoder.pt"
            export_decoder_artifact(
                str(path),
                decoder_state_dict=decoder.state_dict(),
                encodec_model="encodec_24khz",
                bandwidth=6.0,
                target_sr=24000,
                step=12,
                run_name="demo",
            )
            payload = load_exported_decoder(
                str(path),
                decoder=decoder,
                expected_encodec_model="encodec_24khz",
                expected_bandwidth=6.0,
            )
            self.assertEqual(payload["artifact_type"], "encodec_decoder")
            self.assertEqual(payload["step"], 12)

    def test_wrong_artifact_type_is_rejected(self):
        decoder = nn.Sequential(nn.Conv1d(8, 8, kernel_size=1))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({"artifact_type": "other", "decoder_state_dict": decoder.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "artifact_type"):
                load_decoder_artifact(str(path))

    def test_wrong_model_is_rejected(self):
        decoder = nn.Sequential(nn.Conv1d(8, 8, kernel_size=1))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decoder.pt"
            export_decoder_artifact(
                str(path),
                decoder_state_dict=decoder.state_dict(),
                encodec_model="encodec_24khz",
                bandwidth=6.0,
                target_sr=24000,
                step=1,
                run_name="demo",
            )
            with self.assertRaisesRegex(ValueError, "model mismatch"):
                load_exported_decoder(
                    str(path),
                    decoder=decoder,
                    expected_encodec_model="other_model",
                    expected_bandwidth=6.0,
                )

    def test_bandwidth_mismatch_is_rejected(self):
        decoder = nn.Sequential(nn.Conv1d(8, 8, kernel_size=1))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decoder.pt"
            export_decoder_artifact(
                str(path),
                decoder_state_dict=decoder.state_dict(),
                encodec_model="encodec_24khz",
                bandwidth=6.0,
                target_sr=24000,
                step=1,
                run_name="demo",
            )
            with self.assertRaisesRegex(ValueError, "bandwidth mismatch"):
                load_exported_decoder(
                    str(path),
                    decoder=decoder,
                    expected_encodec_model="encodec_24khz",
                    expected_bandwidth=3.0,
                )

    def test_decoder_shape_mismatch_is_rejected(self):
        source_decoder = nn.Sequential(nn.Conv1d(8, 8, kernel_size=1))
        target_decoder = nn.Sequential(nn.Conv1d(4, 4, kernel_size=1))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "decoder.pt"
            export_decoder_artifact(
                str(path),
                decoder_state_dict=source_decoder.state_dict(),
                encodec_model="encodec_24khz",
                bandwidth=6.0,
                target_sr=24000,
                step=1,
                run_name="demo",
            )
            with self.assertRaisesRegex(ValueError, "mismatched_shapes"):
                load_exported_decoder(
                    str(path),
                    decoder=target_decoder,
                    expected_encodec_model="encodec_24khz",
                    expected_bandwidth=6.0,
                )


if __name__ == "__main__":
    unittest.main()
