#!/usr/bin/env python3
"""Export small, real-gap audio-fill examples for every run in every phase.

For each run, loads its "latest" checkpoint (the last one actually trained -- `final.pt` for a
normally-completed run, `latest.pt` for the two manually-killed original baseline/bigmodel runs,
which have no `final.pt`) and its `best_val.pt` checkpoint (lowest `val/combined_loss`), then
runs the model's own real-gap inpainting routine (`Trainer.inpaint_all_gaps`, the same code path
used for the periodic `test_fill_every` training-time snapshots) once per checkpoint. This
produces one small WAV crop per real gap (all 4 carved gaps in the dataset -- 15/20/30/50ms, not
3), decoded exactly around the gap+context window, named `gap{i}_step_{N}.wav` where N is the
checkpoint's own step.

Reuses `phase_0/common.py`'s Trainer-reconstruction (`build_trainer`) -- loading never touches
`outputs/runs/**`, everything is redirected to phase_0's scratch space -- then the resulting wavs
are copied into the correct phase's `results/audio_samples/<run_name>/{latest,best_val}/`
folder, which is (deliberately) NOT covered by the general `phase_[0-9]*/results/**/*.wav`
gitignore rule -- see the `!phase_[0-9]*/results/audio_samples/**/*.wav` exception added
alongside this script.

Usage:
    uv run python scripts/export_gap_audio_samples.py
    uv run python scripts/export_gap_audio_samples.py --runs phase2_regularization
"""
import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "phase_0"))
import common as c  # noqa: E402

# (run_name, phase_folder, checkpoint tag to use for "latest") -- "latest" tag varies because
# baseline/bigmodel were manually killed and have no final.pt (see phase_0/common.py's RUNS).
RUN_TABLE = [
    ("baseline", "phase_0", "latest"),
    ("bigmodel", "phase_0", "latest"),
    ("phase1_songs01", "phase_1", "final"),
    ("phase1_songs04", "phase_1", "final"),
    ("phase1_songs17", "phase_1", "final"),
    ("phase2_regularization", "phase_2", "final"),
    ("phase3_songs17_seed43", "phase_3", "final"),
    ("phase3_songs17_seed44", "phase_3", "final"),
    ("phase3_regularization_seed43", "phase_3", "final"),
    ("phase3_regularization_seed44", "phase_3", "final"),
]


def export_one(run_name: str, phase_folder: str, latest_tag: str) -> None:
    for tag_label, checkpoint_tag in (("latest", latest_tag), ("best_val", "best_val")):
        trainer = c.build_trainer(run_name, checkpoint_tag)
        scratch_samples_dir = trainer.cfg.samples_dir
        trainer.inpaint_all_gaps(save_crop_samples=True)

        dest_dir = REPO_ROOT / phase_folder / "results" / "audio_samples" / run_name / tag_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(scratch_samples_dir.glob(f"gap*_step_{trainer.global_step}.wav"))
        if not wavs:
            raise RuntimeError(f"No gap wavs produced for {run_name}/{checkpoint_tag} in {scratch_samples_dir}")
        for wav in wavs:
            shutil.copyfile(wav, dest_dir / wav.name)
        print(f"{run_name:32s} {tag_label:9s} (step {trainer.global_step:>8d}) -> {dest_dir} ({len(wavs)} gaps)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=[r[0] for r in RUN_TABLE], choices=[r[0] for r in RUN_TABLE])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    table = {name: (phase_folder, latest_tag) for name, phase_folder, latest_tag in RUN_TABLE}
    for run_name in args.runs:
        phase_folder, latest_tag = table[run_name]
        export_one(run_name, phase_folder, latest_tag)


if __name__ == "__main__":
    main()
