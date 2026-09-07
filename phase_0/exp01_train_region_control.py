#!/usr/bin/env python3
"""
Phase 0 / Experiment 1 -- Train-region control (the decisive check).

Question: is the validation pipeline itself trustworthy, or is the ~1% top-1 accuracy we saw
on held-out regions an artifact of a bug in how validation examples are built?

Method: for each usable checkpoint, rebuild the validation-example machinery but force region
selection to avoid the run's real validation holdout ranges -- i.e. sample from territory the
training dataset was allowed to draw from (see common.build_train_region_control_examples).
Run the identical Trainer.run_validation() code path on these "control" examples and compare
against (a) the real holdout-region validation, reproduced fresh, and (b) the training loop's
own contemporaneous train/acc_top1 (read from each run's TensorBoard event file).

If control accuracy tracks the training loop's own reported train accuracy at that step while
holdout accuracy stays near-random: the pipeline is sound and the generalization gap is real.
If control accuracy is close to holdout accuracy instead: validation construction has a bug and
every previous result is void.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    EventAccumulator = None


def read_train_acc_at_step(run_name: str, step: int) -> float:
    """Nearest logged train/acc_top1 to `step` from the real run's TensorBoard file (not the
    Phase 0 scratch writer) -- ground truth for what the training loop itself reported."""
    if EventAccumulator is None:
        return float("nan")
    tb_dir = c.REPO_ROOT / "outputs/runs/nuvole_bianche_short_gaps" / (
        "nuvole_bianche_short_gaps_encoder_decoder" if run_name == "baseline"
        else "nuvole_bianche_short_gaps_encoder_decoder_bigmodel"
    ) / "tb"
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars("train/acc_top1")
    nearest = min(events, key=lambda e: abs(e.step - step))
    return float(nearest.value)


def run_one(run_name: str, checkpoint_tag: str) -> list:
    trainer = c.build_trainer(run_name, checkpoint_tag)
    step = trainer.global_step
    rows = []

    # (a) real holdout-region validation, reproduced fresh
    real_examples, holdout_ranges = c.build_real_validation_examples(trainer)
    dls, specs = c.build_validation_dataloaders(trainer, real_examples)
    holdout_result = c.run_validation_safe(trainer, dls, specs, step=0)
    holdout_scalars = trainer.writer.as_dict()

    # (b) train-region control -- same construction, but regions forced away from holdout_ranges
    trainer.writer = c.RecordingWriter()
    control_examples = c.build_train_region_control_examples(trainer)
    dls2, specs2 = c.build_validation_dataloaders(trainer, control_examples)
    control_result = c.run_validation_safe(trainer, dls2, specs2, step=0)
    control_scalars = trainer.writer.as_dict()

    # (c) training loop's own reported train accuracy at this step, from the real run's TB file
    train_acc = read_train_acc_at_step(run_name, step)

    print(f"\n=== {run_name} / {checkpoint_tag} (step {step}) ===")
    print(f"  training loop's own train/acc_top1 @ step {step}:  {train_acc:.4f}")
    print(f"  HOLDOUT  high_acc_top1={holdout_scalars.get('val/high_acc_top1'):.4f}  low_acc_top1={holdout_scalars.get('val/low_acc_top1'):.4f}  combined_loss={holdout_result['combined_loss']:.4f}")
    print(f"  CONTROL  high_acc_top1={control_scalars.get('val/high_acc_top1'):.4f}  low_acc_top1={control_scalars.get('val/low_acc_top1'):.4f}  combined_loss={control_result['combined_loss']:.4f}")

    record = {
        "run": run_name,
        "checkpoint": checkpoint_tag,
        "step": step,
        "train_acc_top1_reported": train_acc,
        "holdout": holdout_scalars,
        "holdout_combined_loss": holdout_result["combined_loss"],
        "holdout_ranges": holdout_ranges,
        "control": control_scalars,
        "control_combined_loss": control_result["combined_loss"],
    }

    for band in ("high", "low"):
        for metric in ("acc_top1", "acc_top5", "loss", "ppl"):
            rows.append({
                "experiment": "exp01_train_region_control",
                "run": run_name, "checkpoint": checkpoint_tag, "step": step,
                "band": band, "mask_len": "", "method": "holdout",
                "metric": metric, "value": holdout_scalars.get(f"val/{band}_{metric}"),
            })
            rows.append({
                "experiment": "exp01_train_region_control",
                "run": run_name, "checkpoint": checkpoint_tag, "step": step,
                "band": band, "mask_len": "", "method": "train_region_control",
                "metric": metric, "value": control_scalars.get(f"val/{band}_{metric}"),
            })
    rows.append({
        "experiment": "exp01_train_region_control",
        "run": run_name, "checkpoint": checkpoint_tag, "step": step,
        "band": "", "mask_len": "", "method": "training_loop", "metric": "acc_top1",
        "value": train_acc,
    })

    return record, rows


def main():
    all_records = []
    all_rows = []
    for run_name, run in c.RUNS.items():
        for checkpoint_tag in run["checkpoints"]:
            record, rows = run_one(run_name, checkpoint_tag)
            all_records.append(record)
            all_rows.extend(rows)

    c.save_json(c.RESULTS_DIR / "exp01_train_region_control.json", all_records)
    c.append_summary_rows(all_rows)
    print(f"\nWrote {c.RESULTS_DIR / 'exp01_train_region_control.json'}")


if __name__ == "__main__":
    main()
