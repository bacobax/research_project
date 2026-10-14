#!/usr/bin/env python3
"""Phase 5 experiment 03_consistency_checks on the Phase 6 model (and the matched songs17 baselines).

Runs the unchanged `phase_5/exp03_consistency_checks.py` with its checkpoints, example set and output folder
swapped (see `p6_protocol.py`); results go to `phase_6/phase5_protocol/results/exp03_consistency_checks/`.

    uv run python phase_6/phase5_protocol/run_exp03.py --device cuda:1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p6_protocol as p  # noqa: E402

p.install()
import exp03_consistency_checks as m  # noqa: E402

m.OUT = p.RESULTS / "exp03_consistency_checks"
m.FIG = m.OUT / "figures"
m.DEFAULT_RUNS = p.RUNS
m.TB_RUNS = {
    "phase6_final_songs17_seed43": "outputs/runs/phase6_final/phase6_final_songs17_seed43/tb",
    "phase3_songs17_seed43": "outputs/runs/phase3_seed_check/phase3_songs17_seed43/tb",
    "phase3_songs17_seed44": "outputs/runs/phase3_seed_check/phase3_songs17_seed44/tb",
}
if __name__ == "__main__":
    m.main()
