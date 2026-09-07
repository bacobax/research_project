#!/usr/bin/env python3
"""
Phase 0 / Experiment 2 -- Token-space baselines vs. the trained model.

Question: does the trained model beat trivial non-learned fillers on the exact same evaluation
examples the real training run used? Token top-1/top-5 accuracy only (audio-domain metrics are
exp03) -- but on the *same* examples and the *same* metric the model was scored on, so this is a
strictly apples-to-apples comparison, independent of whether that metric is a good proxy for
audio quality (that question is what exp03 addresses).

Baselines (common.BASELINE_NAMES): repeat_left, repeat_right, nearest, random_tokens,
codec_ceiling (ground truth, unmasked -- not a filling method, the best score achievable in this
token space). random_tokens should land near the 1/1024 = 0.0977% floor; codec_ceiling should be
100% by construction (sanity checks).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402


def run_one(run_name: str, checkpoint_tag: str) -> list:
    trainer = c.build_trainer(run_name, checkpoint_tag)
    step = trainer.global_step
    rng = np.random.default_rng(1234)

    # The exact same examples the real training run evaluated on (deterministic given seed).
    grouped_examples, _ = c.build_real_validation_examples(trainer)
    per_length_groups = {k: v for k, v in grouped_examples.items() if "_len_" in k}

    rows = []
    records = []
    for group_name, examples in per_length_groups.items():
        band = "high_activity" if group_name.startswith("high_activity") else "low_activity"
        mask_len = int(group_name.rsplit("_", 1)[-1])

        method_correct = {name: [] for name in c.BASELINE_NAMES}
        method_correct["model"] = []
        model_top5_correct = []

        for example in examples:
            for name in c.BASELINE_NAMES:
                filled = c.fill_baseline(name, example, rng)
                acc = c.token_accuracy(filled, example)
                method_correct[name].append(acc["acc_top1"])

            filled_model = trainer._predict_validation_window(example)
            acc_model = c.token_accuracy(filled_model, example)
            method_correct["model"].append(acc_model["acc_top1"])

        for method, accs in method_correct.items():
            mean_acc = float(np.mean(accs))
            records.append({
                "run": run_name, "checkpoint": checkpoint_tag, "step": step,
                "group": group_name, "band": band, "mask_len": mask_len,
                "method": method, "acc_top1": mean_acc, "n_examples": len(accs),
            })
            rows.append({
                "experiment": "exp02_baselines", "run": run_name, "checkpoint": checkpoint_tag,
                "step": step, "band": band, "mask_len": mask_len, "method": method,
                "metric": "acc_top1", "value": mean_acc,
            })

    print(f"\n=== {run_name} / {checkpoint_tag} (step {step}) ===")
    by_method = {}
    for r in records:
        by_method.setdefault(r["method"], []).append(r["acc_top1"])
    for method, accs in by_method.items():
        print(f"  {method:16s} mean acc_top1 over all bands/lengths = {np.mean(accs):.4f}")

    return records, rows


def main():
    all_records = []
    all_rows = []
    for run_name, run in c.RUNS.items():
        for checkpoint_tag in run["checkpoints"]:
            records, rows = run_one(run_name, checkpoint_tag)
            all_records.extend(records)
            all_rows.extend(rows)

    c.save_json(c.RESULTS_DIR / "exp02_baselines.json", all_records)
    c.append_summary_rows(all_rows)
    print(f"\nWrote {c.RESULTS_DIR / 'exp02_baselines.json'}")

    # Sanity checks
    random_accs = [r["acc_top1"] for r in all_records if r["method"] == "random_tokens"]
    ceiling_accs = [r["acc_top1"] for r in all_records if r["method"] == "codec_ceiling"]
    print(f"\nSanity: random_tokens mean acc_top1 = {np.mean(random_accs):.5f} (expect ~0.00098)")
    print(f"Sanity: codec_ceiling mean acc_top1 = {np.mean(ceiling_accs):.5f} (expect 1.0)")


if __name__ == "__main__":
    main()
