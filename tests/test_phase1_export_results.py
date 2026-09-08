import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from phase_1.export_results import export_results, load_scalar_points


def _write_event_file(tb_dir: Path, *, suffix: str, combined_values, walltime_offset=0.0):
    writer = SummaryWriter(log_dir=str(tb_dir), filename_suffix=suffix)
    writer.add_scalar("model/params_M", 1.25, 0, walltime=100.0 + walltime_offset)
    writer.add_scalar("dataset/total_minutes", 2.0, 0, walltime=100.0 + walltime_offset)
    writer.add_scalar("train/avg_loss", 6.0, 1000, walltime=101.0 + walltime_offset)
    for step, combined in combined_values:
        walltime = 100.0 + step + walltime_offset
        writer.add_scalar("val/combined_loss", combined, step, walltime=walltime)
        writer.add_scalar("val/high_acc_top1", 0.20 + step / 100000.0, step, walltime=walltime)
        writer.add_scalar("val/low_acc_top1", 0.10 + step / 100000.0, step, walltime=walltime)
        for band in ("high", "low"):
            for length in (1, 2):
                base = combined + (0.1 if band == "low" else 0.0) + length / 100.0
                metrics = {
                    "loss": base,
                    "nll": base,
                    "ppl": 20.0 + length,
                    "acc_top1": 0.2 + length / 100.0,
                    "acc_top5": 0.5 + length / 100.0,
                }
                for metric, value in metrics.items():
                    writer.add_scalar(
                        f"val/{band}_len_{length}_{metric}", value, step, walltime=walltime
                    )
    writer.close()


def _make_fake_repo(root: Path, run_names=("phase1_songs01",)) -> None:
    for relative in (
        "src/audio_infill/train.py",
        "src/audio_infill/config.py",
        "pyproject.toml",
        "uv.lock",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")

    for index, run_name in enumerate(run_names, start=1):
        config = root / "configs" / "train" / f"{run_name}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"run_name: {run_name}\ntoken_frame_rate: 50\n", encoding="utf-8")

        run_dir = root / "outputs" / "runs" / "phase1_data_scaling" / run_name
        (run_dir / "tb").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        _write_event_file(
            run_dir / "tb",
            suffix=f".{index}",
            combined_values=((10, 2.0), (20, 1.5), (30, 1.8)),
        )
        (run_dir / "songs.json").write_text(
            '{"pivot":{"duration_s":60,"frames":100},'
            '"extra_songs":[{"duration_s":30,"frames":50}],"song_sampling":"duration"}\n',
            encoding="utf-8",
        )
        (run_dir / "launch.log").write_text(
            "2026-01-01 10:00:00 [INFO] infiller: Device: cpu\n"
            "2026-01-01 10:10:00 [WARNING] infiller: Early stopping at step 30: test\n"
            f"2026-01-01 10:10:01 [INFO] infiller: Saved checkpoint: "
            f"{run_dir}/checkpoints/final.pt (step 30)\n",
            encoding="utf-8",
        )
        torch.save({"step": 20}, run_dir / "checkpoints" / "best_val.pt")
        torch.save({"step": 30}, run_dir / "checkpoints" / "final.pt")


def _tree_hashes(root: Path):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class TensorBoardExtractionTests(unittest.TestCase):
    def test_duplicate_tag_step_uses_latest_wall_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            tb_dir = Path(tmp)
            _write_event_file(tb_dir, suffix=".old", combined_values=((10, 9.0),))
            _write_event_file(
                tb_dir,
                suffix=".new",
                combined_values=((10, 3.0),),
                walltime_offset=1000.0,
            )
            points = load_scalar_points(tb_dir)
            combined = [point for point in points if point.tag == "val/combined_loss"]
            self.assertEqual(len(combined), 1)
            self.assertEqual(combined[0].step, 10)
            self.assertAlmostEqual(combined[0].value, 3.0)


class Phase1ExportTests(unittest.TestCase):
    def test_export_writes_tables_figures_and_is_deterministic_without_qualitative_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_names = ("phase1_songs01", "phase1_songs04")
            _make_fake_repo(root, run_names)
            kwargs = {
                "repo_root": root,
                "run_names": run_names,
                "input_root": Path("outputs/runs/phase1_data_scaling"),
                "output_root": Path("phase_1/results"),
                "command": "fixture export",
            }
            runs = export_results(**kwargs)
            output = root / "phase_1" / "results"
            self.assertEqual([run.best_step for run in runs], [20, 20])
            self.assertEqual(runs[0].summary["stop_step"], 30)
            self.assertEqual(runs[0].summary["stop_reason"], "early_stopping")
            self.assertEqual(runs[0].groups[0]["gap_ms"], 20.0)
            for relative in (
                "metrics.csv",
                "run_summary.csv",
                "best_by_group.csv",
                "tables/run_summary.tex",
                "tables/best_by_group.tex",
                "figures/validation_loss.pdf",
                "figures/validation_loss.png",
                "figures/validation_accuracy.pdf",
                "figures/training_loss.pdf",
                "figures/best_by_gap_length.pdf",
                "provenance.json",
                "README.md",
            ):
                self.assertGreater((output / relative).stat().st_size, 0, relative)
            self.assertFalse(any(output.rglob("*.pt")))
            self.assertFalse(any(output.rglob("*.wav")))
            self.assertFalse(any(output.rglob("events.out.tfevents.*")))
            before = _tree_hashes(output)
            export_results(**kwargs)
            self.assertEqual(before, _tree_hashes(output))

    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_repo(root)
            (root / "outputs/runs/phase1_data_scaling/phase1_songs01/checkpoints/final.pt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                export_results(
                    repo_root=root,
                    run_names=("phase1_songs01",),
                    input_root=Path("outputs/runs/phase1_data_scaling"),
                    output_root=Path("phase_1/results"),
                    command="fixture export",
                )


if __name__ == "__main__":
    unittest.main()
