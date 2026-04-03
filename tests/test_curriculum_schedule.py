import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.config import TrainConfig
from audio_infill.train import Trainer


class TestCurriculumSchedule(unittest.TestCase):
    def test_curriculum_reaches_max_at_coverage_step(self):
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = TrainConfig(
            curriculum=True,
            curriculum_start_mask=128,
            curriculum_end_mask=750,
            curriculum_warmup_frac=0.1,
            curriculum_coverage=0.5,
            total_steps=100,
            mask_len_min=187,
            mask_len_max=750,
        )
        trainer.largest_gap_frames = 750
        trainer.dataset = mock.Mock()
        trainer.writer = mock.Mock()

        trainer._init_curriculum()

        self.assertEqual(trainer.curriculum_warmup_steps, 10)
        self.assertEqual(trainer.curriculum_reach_max_step, 50)

        _, before_max, progress_before = trainer._curriculum_update(49)
        self.assertLess(before_max, 750)
        self.assertLess(progress_before, 1.0)

        min_at_max, max_at_max, progress_at_max = trainer._curriculum_update(50)
        self.assertEqual(progress_at_max, 1.0)
        self.assertEqual(min_at_max, 187)
        self.assertEqual(max_at_max, 750)


if __name__ == "__main__":
    unittest.main()
