"""Progress report for the Phase 6 run (reads TensorBoard scalars + checkpoints; no GPU use).

    uv run python phase_6/monitor.py                 # one report
    watch -n 300 'uv run python phase_6/monitor.py'  # refresh every 5 min
"""
import argparse
import glob
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/runs/phase6_final/phase6_final_songs17_seed43"
TOTAL = 300_000
BASE = {  # songs17 seed family @300k (phase_3/results/seed_check): mean +- std
    "val/combined_loss": (4.9569, 0.0568), "val/si_sdr_db_gap_only": (-4.49, 2.96),
}


def scalars():
    """Read all TensorBoard scalar events of the run into {tag: [(step, value, wall_time)]}."""
    out = {}
    for f in sorted(glob.glob(str(RUN / "tb" / "events.out.tfevents.*"))):
        for ev in EventFileLoader(f).Load():
            for v in ev.summary.value:
                kind = v.WhichOneof("value")
                if kind == "simple_value":
                    val = v.simple_value
                elif kind == "tensor" and v.tensor.dtype == 1 and len(v.tensor.float_val) == 1:  # DT_FLOAT scalar
                    val = v.tensor.float_val[0]
                elif kind == "tensor" and v.tensor.tensor_content and len(v.tensor.tensor_content) == 4:
                    val = struct.unpack("<f", v.tensor.tensor_content)[0]
                else:
                    continue
                out.setdefault(v.tag, []).append((ev.step, val, ev.wall_time))
    return out


def main():
    """Print the progress report (step/ETA, latest train metrics, recent validation rows versus the baseline family) for the Phase 6 run."""
    argparse.ArgumentParser().parse_args()
    s = scalars()
    if "train/avg_nll" not in s:
        print("no training scalars yet (still loading/encoding songs?)")
    else:
        step, _, t_now = s["train/avg_nll"][-1]
        sps = s["train/steps_per_sec"][-1][1]
        t_first = s["train/avg_nll"][0][2]
        elapsed_h = (t_now - t_first) / 3600
        eta_h = (TOTAL - step) / max(sps, 1e-9) / 3600
        age_min = (time.time() - t_now) / 60
        print(f"step {step:,}/{TOTAL:,} ({100*step/TOTAL:.1f}%)  {sps:.2f} it/s  elapsed {elapsed_h:.1f} h  ETA {eta_h:.1f} h  (last log {age_min:.0f} min ago)")
        for tag in ("train/avg_nll", "train/avg_acc_top1", "train/grad_norm_avg", "train/lr"):
            if tag in s:
                print(f"  {tag:24s} {s[tag][-1][1]:.4g}")
    print("\nvalidation (every 10k steps; zeroed-context primary set, fp32):")
    keys = ["val/combined_loss", "val_clean/combined_loss", "val/high_acc_top1", "val/low_acc_top1", "val/si_sdr_db", "val/si_sdr_db_gap_only"]
    if "val/combined_loss" in s:
        print("  step     " + " ".join(f"{k.split('/')[-1][:14]:>14s}" for k in keys))
        steps = [st for st, _, _ in s["val/combined_loss"]]
        for st in steps[-12:]:
            row = []
            for k in keys:
                d = {a: b for a, b, _ in s.get(k, [])}
                row.append(f"{d[st]:14.4f}" if st in d else f"{'-':>14s}")
            print(f"  {st:>7,} " + " ".join(row))
        best = min(s["val/combined_loss"], key=lambda x: x[1])
        print(f"  best val NLL {best[1]:.4f} @ {best[0]:,}   (baseline family @300k: {BASE['val/combined_loss'][0]} +- {BASE['val/combined_loss'][1]})")
        if "val/si_sdr_db_gap_only" in s:
            print(f"  latest gap-only SI-SDR {s['val/si_sdr_db_gap_only'][-1][1]:.2f} dB   (baseline family: {BASE['val/si_sdr_db_gap_only'][0]} +- {BASE['val/si_sdr_db_gap_only'][1]})")
    else:
        print("  none yet")
    print("\ncheckpoints:")
    for p in sorted(glob.glob(str(RUN / "checkpoints" / "*.pt"))):
        print(f"  {os.path.basename(p):16s} {os.path.getsize(p)/1e6:5.0f} MB  {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p)))}")
    free = shutil.disk_usage("/").free / 1e9
    print(f"\ndisk free {free:.1f} GB")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader", "-i", "1"],
                             capture_output=True, text=True).stdout.strip()
        print(f"gpu1: {gpu}")
    except Exception:
        pass
    alive = subprocess.run(["pgrep", "-f", "phase6_final_songs17_seed43.yaml"], capture_output=True, text=True).stdout.split()
    print("process:", "running" if alive else "NOT RUNNING")
    log = REPO / "outputs/runs/phase6_final/train.log"
    bad = [l for l in log.read_text(errors="ignore").splitlines() if "non-finite" in l or "Traceback" in l or "Early stopping" in l]
    if bad:
        print("!! log alerts:", *bad[-3:], sep="\n   ")


if __name__ == "__main__":
    main()
