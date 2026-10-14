#!/usr/bin/env python3
"""Phase 5 experiment 01_context_contamination on the Phase 6 model (and the matched songs17 baselines).

Runs the unchanged `phase_5/exp01_context_contamination.py` with its checkpoints, example set and output folder
swapped (see `p6_protocol.py`); results go to `phase_6/phase5_protocol/results/exp01_context_contamination/`.

    uv run python phase_6/phase5_protocol/run_exp01.py --device cuda:1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p6_protocol as p  # noqa: E402

p.install()
import exp01_context_contamination as m  # noqa: E402

m.OUT = p.RESULTS / "exp01_context_contamination"
m.FIG = m.OUT / "figures"
m.DEFAULT_RUNS = p.RUNS
m.score = p.score_exp01
if __name__ == "__main__":
    m.main()
