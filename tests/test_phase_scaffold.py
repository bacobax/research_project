import tempfile
import unittest
from pathlib import Path

from scripts.scaffold_phase import RESULT_SUBDIRECTORIES, scaffold_phase


class PhaseScaffoldTests(unittest.TestCase):
    def test_scaffold_creates_raw_and_tracked_result_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir, results_dir = scaffold_phase(root, 2)

            self.assertEqual(raw_dir, root / "outputs/runs/phase_2")
            self.assertEqual(results_dir, root / "phase_2/results")
            self.assertTrue(raw_dir.is_dir())
            self.assertTrue((root / "phase_2/README.md").is_file())
            self.assertTrue((results_dir / "README.md").is_file())
            for subdirectory in RESULT_SUBDIRECTORIES:
                self.assertTrue((results_dir / subdirectory / ".gitkeep").is_file())

    def test_scaffold_is_idempotent_and_preserves_existing_documentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, results_dir = scaffold_phase(root, 3)
            readme = results_dir / "README.md"
            readme.write_text("custom documentation\n", encoding="utf-8")

            scaffold_phase(root, 3)

            self.assertEqual(readme.read_text(encoding="utf-8"), "custom documentation\n")

    def test_negative_phase_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "non-negative"):
                scaffold_phase(Path(tmp), -1)


if __name__ == "__main__":
    unittest.main()
