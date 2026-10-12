import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import EncoderDecoderCodebookInfiller

K, BINS, T = 4, 16, 24


def make_model(conditioning: str) -> EncoderDecoderCodebookInfiller:
    torch.manual_seed(0)
    model = EncoderDecoderCodebookInfiller(
        K=K, bins=BINS, mask_token=BINS, d_model=32, n_heads=4, n_layers=1, max_len=64,
        dropout=0.0, codebook_conditioning=conditioning, depth_layers=2,
    )
    return model.eval()


def make_batch():
    g = torch.Generator().manual_seed(1)
    y = torch.randint(0, BINS, (2, K, T), generator=g)
    x = y.clone()
    x[0, :, 5:8] = BINS
    x[1, :, 10:12] = BINS
    return x, y


class TestCodebookConditioning(unittest.TestCase):
    def test_none_has_no_depth_parameters(self):
        model = make_model("none")
        self.assertFalse(any(k.startswith("depth") for k in model.state_dict()))
        x, y = make_batch()
        self.assertEqual(tuple(model(x).shape), (2, K, T, BINS))
        # targets are ignored when heads are independent
        self.assertTrue(torch.equal(model(x), model(x, gap_targets=y)))

    def test_shapes_and_zero_outside_gap(self):
        model = make_model("depth_ar")
        x, y = make_batch()
        with torch.no_grad():
            out_tf = model(x, gap_targets=y)
            out_g = model(x)
        self.assertEqual(tuple(out_tf.shape), (2, K, T, BINS))
        self.assertEqual(tuple(out_g.shape), (2, K, T, BINS))
        self.assertEqual(float(out_g[0, :, :5].abs().sum()), 0.0)
        self.assertEqual(float(out_g[0, :, 8:].abs().sum()), 0.0)

    def test_teacher_forcing_is_causal_over_codebooks(self):
        model = make_model("depth_ar")
        x, y = make_batch()
        y2 = y.clone()
        y2[:, 2:, :] = (y2[:, 2:, :] + 1) % BINS  # change codebooks >= 2
        with torch.no_grad():
            a = model(x, gap_targets=y)
            b = model(x, gap_targets=y2)
        # logits of codebooks 0..2 only see tokens of codebooks < k, so 0..2 are unchanged
        self.assertTrue(torch.allclose(a[:, :3], b[:, :3], atol=1e-6))
        self.assertFalse(torch.allclose(a[:, 3:], b[:, 3:], atol=1e-6))

    def test_greedy_logits_equal_teacher_forcing_on_greedy_tokens(self):
        model = make_model("depth_ar")
        x, y = make_batch()
        with torch.no_grad():
            greedy = model(x)
            tokens = greedy.argmax(dim=-1)
            tf = model(x, gap_targets=tokens)
        gap = torch.zeros(2, T, dtype=torch.bool)
        gap[0, 5:8] = True
        gap[1, 10:12] = True
        for b in range(2):
            self.assertTrue(torch.allclose(greedy[b][:, gap[b]], tf[b][:, gap[b]], atol=1e-5))

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            make_model("bogus")


if __name__ == "__main__":
    unittest.main()
