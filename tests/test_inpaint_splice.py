import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import reallike_span, splice_gap_audio, touched_frames


class TestSplice(unittest.TestCase):
    def setUp(self):
        self.base = np.linspace(-1, 1, 2000, dtype=np.float32)
        self.base[900:1000] = 0.0  # the gap
        self.decoded = np.full(800, 0.5, dtype=np.float32)  # covers samples 600..1400

    def test_only_gap_and_fade_change(self):
        out = splice_gap_audio(self.base, self.decoded, 600, 900, 1000, 40)
        np.testing.assert_array_equal(out[:860], self.base[:860])
        np.testing.assert_array_equal(out[1040:], self.base[1040:])
        np.testing.assert_allclose(out[900:1000], 0.5)
        self.assertFalse(np.array_equal(out[860:900], self.base[860:900]))

    def test_fade_is_monotonic_blend(self):
        out = splice_gap_audio(self.base, self.decoded, 600, 900, 1000, 40)
        w = (out[860:900] - self.base[860:900]) / (0.5 - self.base[860:900])
        self.assertTrue(np.all(np.diff(w) > 0))
        self.assertTrue(np.all((w >= 0) & (w <= 1)))

    def test_no_fade_replaces_exactly_the_gap(self):
        out = splice_gap_audio(self.base, self.decoded, 600, 900, 1000, 0)
        np.testing.assert_array_equal(out[:900], self.base[:900])
        np.testing.assert_array_equal(out[1000:], self.base[1000:])
        np.testing.assert_allclose(out[900:1000], 0.5)

    def test_input_not_modified(self):
        before = self.base.copy()
        splice_gap_audio(self.base, self.decoded, 600, 900, 1000, 40)
        np.testing.assert_array_equal(self.base, before)


class TestZeroSpan(unittest.TestCase):
    def test_reallike_span_rounds_back_to_mask(self):
        hop = 320
        rng = np.random.default_rng(0)
        for g0, g1 in [(100, 101), (500, 504), (7, 9)]:
            for r in (None, rng):
                s, e = reallike_span(g0, g1, hop, r)
                self.assertEqual((int(round(s / hop)), int(round(e / hop))), (g0, g1))
                t0, t1 = touched_frames(s, e, hop)
                self.assertLessEqual(t0, g0)
                self.assertGreaterEqual(t1, g1)

    def test_deterministic_without_rng(self):
        self.assertEqual(reallike_span(100, 103, 320), reallike_span(100, 103, 320))


if __name__ == "__main__":
    unittest.main()
