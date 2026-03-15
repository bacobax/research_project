import unittest
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestDatasetLogic(unittest.TestCase):
    def test_single_gap_annotation_consistency(self):
        ann_path = Path("data/processed/single_gap/wav_test_gap_5p000s.json")
        with ann_path.open("r", encoding="utf-8") as f:
            ann = json.load(f)

        self.assertIn("gap", ann)
        gap = ann["gap"]
        self.assertGreater(gap["gap_end_s"], gap["gap_start_s"])
        self.assertEqual(gap["gap_end_sample"] - gap["gap_start_sample"], gap["gap_len_samples"])

    def test_multigap_annotation_non_overlapping(self):
        ann_path = Path("data/processed/multigap/wav_test_multigap_4x_1p0s_2p0s_5p0s_10p0s.json")
        with ann_path.open("r", encoding="utf-8") as f:
            ann = json.load(f)

        gaps = sorted(ann["gaps"], key=lambda g: g["gap_start_s"])
        self.assertEqual(len(gaps), ann["num_gaps"])
        min_sep = ann["min_gap_separation_seconds"]
        for i in range(1, len(gaps)):
            self.assertGreaterEqual(gaps[i]["gap_start_s"] - gaps[i - 1]["gap_end_s"], min_sep - 1e-6)


if __name__ == "__main__":
    unittest.main()
